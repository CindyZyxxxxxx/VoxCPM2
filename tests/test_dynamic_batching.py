import pytest

from voxcpm.training.data import (
    BufferedDynamicBatchSampler,
    DynamicBatchSampler,
    SampleLength,
    compute_sample_lengths,
    select_prompt_audio,
)


def _flatten(batches):
    return [index for batch in batches for index in batch]


def _length(packed, *, target=None, ref=0, text=0):
    target = packed - ref - text - (4 if ref else 2) if target is None else target
    return SampleLength(
        text_tokens=text,
        target_samples=target * 640,
        target_sequence=target,
        ref_samples=ref * 640,
        ref_sequence=ref,
        packed=packed,
    )


def test_dynamic_batches_respect_token_and_sample_limits():
    lengths = [_length(length, target=length) for length in [2, 3, 4, 5, 6, 7]]
    sampler = DynamicBatchSampler(
        lengths,
        max_batch_tokens=12,
        batch_size=4,
        shuffle=False,
    )

    batches = list(sampler)

    assert sorted(_flatten(batches)) == list(range(len(lengths)))
    assert all(len(batch) <= 4 for batch in batches)
    assert all(sampler._padded_token_cost(batch) <= 12 for batch in batches)
    assert len(sampler) == len(batches)


def test_dynamic_batches_shuffle_deterministically_by_epoch():
    sampler = DynamicBatchSampler(
        [_length(length, target=length) for length in range(1, 41)],
        max_batch_tokens=100,
        batch_size=4,
        seed=7,
    )

    epoch_zero = list(sampler)
    assert epoch_zero == list(sampler)

    sampler.set_epoch(1)
    epoch_one = list(sampler)
    assert epoch_one == list(sampler)
    assert epoch_one != epoch_zero
    assert sorted(_flatten(epoch_one)) == list(range(40))


def test_dynamic_batches_are_evenly_sharded_across_ranks():
    lengths = [_length(5, target=5) for _ in range(25)]
    samplers = [
        DynamicBatchSampler(
            lengths,
            max_batch_tokens=20,
            batch_size=4,
            seed=11,
            rank=rank,
            world_size=3,
        )
        for rank in range(3)
    ]

    rank_batches = [list(sampler) for sampler in samplers]

    assert len({len(batches) for batches in rank_batches}) == 1
    assert all(len(sampler) == len(rank_batches[rank]) for rank, sampler in enumerate(samplers))
    assert all(
        sampler._padded_token_cost(batch) <= 20 for sampler, batches in zip(samplers, rank_batches) for batch in batches
    )


def test_complementary_target_and_ref_lengths_do_not_create_high_padding_batch():
    lengths = [
        _length(14, target=8, ref=2),
        _length(14, target=2, ref=8),
        _length(14, target=8, ref=2),
        _length(14, target=2, ref=8),
    ]
    sampler = DynamicBatchSampler(
        lengths,
        max_batch_tokens=28,
        batch_size=2,
        shuffle=False,
        bucket_size=4,
    )

    batches = list(sampler)

    assert sorted(_flatten(batches)) == list(range(4))
    assert all(sampler._padded_token_cost(batch) <= 28 for batch in batches)
    assert all(len({lengths[index].target_sequence for index in batch}) == 1 for batch in batches)


def test_reference_presence_is_a_primary_bucket_boundary():
    lengths = [
        _length(10, target=8),
        _length(10, target=4, ref=2),
        _length(10, target=8),
        _length(10, target=4, ref=2),
    ]
    sampler = DynamicBatchSampler(lengths, max_batch_tokens=40, batch_size=4, shuffle=False)

    assert all(len({lengths[index].has_ref_audio for index in batch}) == 1 for batch in sampler)


def test_compute_sample_lengths_returns_structured_audio_lengths():
    class FakeDataset:
        column_names = ["text_ids", "audio", "ref_audio"]

        def __init__(self):
            self.items = [
                {
                    "text_ids": [1, 2, 3],
                    "audio": {"array": [0.0] * 1600, "sampling_rate": 16000},
                    "ref_audio": {"array": [0.0] * 800, "sampling_rate": 16000},
                },
                {
                    "text_ids": [1],
                    "audio": {"array": [0.0] * 3200, "sampling_rate": 16000},
                    "ref_audio": None,
                },
            ]

        def __len__(self):
            return len(self.items)

        def __getitem__(self, key):
            if isinstance(key, str):
                return [item[key] for item in self.items]
            return self.items[key]

        def __iter__(self):
            return iter(self.items)

    lengths = compute_sample_lengths(FakeDataset(), audio_vae_fps=20, patch_size=2)

    assert lengths[0] == SampleLength(3, 1600, 1, 800, 1, 9)
    assert lengths[1] == SampleLength(1, 3200, 2, 0, 0, 5)


def test_select_prompt_audio_by_highest_ssim():
    candidates = [
        {"audio": "first.wav", "ssim": 0.7},
        {"audio": "best.wav", "ssim": 0.95},
        {"audio": "third.wav", "ssim": 0.8},
    ]

    assert select_prompt_audio(candidates, "highest_ssim") == "best.wav"


def test_select_prompt_audio_random_is_deterministic_per_row():
    candidates = [{"audio": f"{index}.wav", "ssim": index / 10} for index in range(5)]

    selected = select_prompt_audio(candidates, "random", seed=123, index=7)

    assert selected == select_prompt_audio(candidates, "random", seed=123, index=7)
    assert selected in {candidate["audio"] for candidate in candidates}


def test_select_shortest_prompt_audio(monkeypatch):
    durations = {"long.wav": 4.0, "short.wav": 1.5, "medium.wav": 2.0}

    class Info:
        def __init__(self, duration):
            self.duration = duration

    import soundfile

    monkeypatch.setattr(soundfile, "info", lambda path: Info(durations[path]))
    candidates = [{"audio": path, "ssim": 0.5} for path in durations]

    assert select_prompt_audio(candidates, "shortest") == "short.wav"


def test_select_prompt_audio_rejects_unknown_strategy():
    with pytest.raises(ValueError, match="expected one of"):
        select_prompt_audio([{"audio": "ref.wav", "ssim": 0.9}], "unknown")


def test_buffered_sampler_measures_only_each_window_lazily():
    rows = [
        {
            "audio": {"array": [0.0] * 1600, "sampling_rate": 16000},
            "ref_audio": None,
            "text_ids": [1, 2],
        }
        for _ in range(8)
    ]
    measured_indices = []

    class FakeDataset:
        def __len__(self):
            return len(rows)

        def __getitem__(self, index):
            measured_indices.append(index)
            return rows[index]

    sampler = BufferedDynamicBatchSampler(
        FakeDataset(),
        audio_vae_fps=25,
        patch_size=1,
        max_batch_tokens=100,
        batch_size=2,
        bucket_size=4,
        shuffle=False,
    )

    iterator = iter(sampler)
    first_batch = next(iterator)

    assert first_batch == [0, 1]
    assert measured_indices == list(range(4))
    assert sampler.progress == 0.5


def test_buffered_sampler_shards_complete_window_batches_across_ranks():
    rows = [{"audio": {"array": [0.0] * 1600, "sampling_rate": 16000}, "text_ids": [1]} for _ in range(12)]

    class FakeDataset:
        def __len__(self):
            return len(rows)

        def __getitem__(self, index):
            return rows[index]

    samplers = [
        BufferedDynamicBatchSampler(
            FakeDataset(),
            audio_vae_fps=25,
            patch_size=1,
            max_batch_tokens=100,
            batch_size=2,
            bucket_size=6,
            rank=rank,
            world_size=2,
            shuffle=False,
            drop_last=True,
        )
        for rank in range(2)
    ]

    batches = [list(sampler) for sampler in samplers]

    assert len(batches[0]) == len(batches[1])
    assert not ({index for batch in batches[0] for index in batch} & {index for batch in batches[1] for index in batch})


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"lengths": [_length(11, target=11)], "max_batch_tokens": 10, "batch_size": 1}, "filtered"),
        ({"lengths": [_length(1, target=1)], "max_batch_tokens": 0, "batch_size": 1}, "greater than zero"),
        ({"lengths": [_length(1, target=1)], "max_batch_tokens": 10, "batch_size": 0}, "greater than zero"),
    ],
)
def test_dynamic_batch_sampler_rejects_invalid_input(kwargs, message):
    with pytest.raises(ValueError, match=message):
        DynamicBatchSampler(**kwargs)
