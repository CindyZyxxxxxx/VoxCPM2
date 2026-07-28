import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import argbind
import torch
from datasets import Audio, Dataset, DatasetDict, load_dataset
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import Sampler

from ..model.voxcpm import VoxCPMConfig
from ..modules.audiovae import AudioVAE
from .packers import AudioFeatureProcessingPacker

DEFAULT_TEXT_COLUMN = "text"
DEFAULT_AUDIO_COLUMN = "audio"
DEFAULT_REF_AUDIO_COLUMN = "ref_audio"
DEFAULT_PROMPT_AUDIO_LIST_COLUMN = "prompt_audio_list"
DEFAULT_ID_COLUMN = "dataset_id"
METADATA_COLUMNS = ("sample_id", "dialect", "language")


@dataclass(frozen=True)
class SampleLength:
    """Lengths used to estimate both packed tokens and collation padding."""

    text_tokens: int
    target_samples: int
    target_sequence: int
    ref_samples: int
    ref_sequence: int
    packed: int

    @property
    def has_ref_audio(self) -> bool:
        return self.ref_samples > 0


def _compute_sample_length(item: Dict, audio_vae_fps: float, patch_size: int) -> SampleLength:
    """Compute one sample's length from decoded target/ref waveforms."""
    text_tokens = len(item["text_ids"])
    audio = item[DEFAULT_AUDIO_COLUMN]
    target_samples = len(audio["array"])
    target_duration = target_samples / float(audio["sampling_rate"])
    target_sequence = math.ceil(math.ceil(target_duration * audio_vae_fps) / patch_size)

    ref_samples = 0
    ref_sequence = 0
    ref_audio = item.get(DEFAULT_REF_AUDIO_COLUMN)
    if ref_audio:
        ref_samples = len(ref_audio["array"])
        ref_duration = ref_samples / float(ref_audio["sampling_rate"])
        ref_sequence = math.ceil(math.ceil(ref_duration * audio_vae_fps) / patch_size)

    packed = text_tokens + target_sequence + ref_sequence + (4 if ref_samples else 2)
    return SampleLength(text_tokens, target_samples, target_sequence, ref_samples, ref_sequence, packed)


class DynamicBatchSampler(Sampler[List[int]]):
    """Build length-aware batches constrained by a padded-token budget.

    Batches are built globally and then sharded across distributed ranks.  This
    keeps the number of batches identical on every rank, which is required by
    DDP.  When ``drop_last`` is false, the final global batch list is padded in
    the same way as :class:`~torch.utils.data.DistributedSampler`.
    """

    def __init__(
        self,
        lengths: List[SampleLength],
        *,
        max_batch_tokens: int,
        batch_size: int,
        seed: int = 42,
        drop_last: bool = False,
        rank: int = 0,
        world_size: int = 1,
        shuffle: bool = True,
        bucket_size: int = 0,
    ):
        if not lengths:
            raise ValueError("DynamicBatchSampler requires at least one sample")
        if max_batch_tokens <= 0:
            raise ValueError("max_batch_tokens must be greater than zero")
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        if world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError(f"Invalid distributed rank/world_size: {rank}/{world_size}")
        if any(length.packed <= 0 for length in lengths):
            raise ValueError("All packed sample lengths must be greater than zero")
        if any(length.packed > max_batch_tokens for length in lengths):
            raise ValueError("Sample lengths must be filtered to max_batch_tokens before batching")

        self.lengths = list(lengths)
        self.max_batch_tokens = max_batch_tokens
        self.batch_size = batch_size
        self.seed = seed
        self.drop_last = drop_last
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.bucket_size = bucket_size if bucket_size > 0 else batch_size * 100
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _global_batches(self) -> List[List[int]]:
        rng = random.Random(self.seed + self.epoch)
        batches: List[List[int]] = []

        # Samples with and without reference audio have different tensor shapes
        # and special-token overhead, so never mix them in one batch.
        ref_groups = {
            False: [i for i, length in enumerate(self.lengths) if not length.has_ref_audio],
            True: [i for i, length in enumerate(self.lengths) if length.has_ref_audio],
        }
        for indices in ref_groups.values():
            if self.shuffle:
                rng.shuffle(indices)
            windows = [indices[start : start + self.bucket_size] for start in range(0, len(indices), self.bucket_size)]
            for window in windows:
                # The initial shuffle randomizes ties, while this two-dimensional
                # ordering keeps target/ref padding similar inside each window.
                window.sort(
                    key=lambda index: (
                        self.lengths[index].target_samples,
                        self.lengths[index].ref_samples,
                        self.lengths[index].packed,
                    )
                )
            ordered_indices = [index for window in windows for index in window]
            current: List[int] = []
            for index in ordered_indices:
                prospective = [*current, index]
                if current and (
                    len(prospective) > self.batch_size or self._padded_token_cost(prospective) > self.max_batch_tokens
                ):
                    batches.append(current)
                    current = []
                current.append(index)
            if current and (not self.drop_last or len(current) == self.batch_size):
                batches.append(current)

        if self.shuffle:
            rng.shuffle(batches)

        remainder = len(batches) % self.world_size
        if remainder:
            if self.drop_last:
                batches = batches[: len(batches) - remainder]
            else:
                missing = self.world_size - remainder
                batches.extend([list(batches[i % len(batches)]) for i in range(missing)])
        return batches

    def _padded_token_cost(self, indices: List[int]) -> int:
        """Estimate packed cost after independently padding each component."""
        lengths = [self.lengths[index] for index in indices]
        count = len(lengths)
        component_cost = count * (
            max(length.text_tokens for length in lengths)
            + max(length.target_sequence for length in lengths)
            + max(length.ref_sequence for length in lengths)
            + (4 if lengths[0].has_ref_audio else 2)
        )
        packed_cost = count * max(length.packed for length in lengths)
        return max(component_cost, packed_cost)

    def __iter__(self):
        batches = self._global_batches()
        yield from batches[self.rank :: self.world_size]

    def __len__(self) -> int:
        return len(self._global_batches()) // self.world_size


class BufferedDynamicBatchSampler(Sampler[List[int]]):
    """Build dynamic batches from bounded, lazily measured shuffle windows."""

    def __init__(
        self,
        dataset: Dataset,
        *,
        audio_vae_fps: float,
        patch_size: int,
        max_batch_tokens: int,
        batch_size: int,
        bucket_size: int,
        seed: int = 42,
        drop_last: bool = False,
        rank: int = 0,
        world_size: int = 1,
        shuffle: bool = True,
    ):
        if max_batch_tokens <= 0 or batch_size <= 0:
            raise ValueError("max_batch_tokens and batch_size must be greater than zero")
        if world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError(f"Invalid distributed rank/world_size: {rank}/{world_size}")
        if bucket_size <= 0:
            bucket_size = batch_size * 100
        self.dataset = dataset
        self.audio_vae_fps = audio_vae_fps
        self.patch_size = patch_size
        self.max_batch_tokens = max_batch_tokens
        self.batch_size = batch_size
        self.bucket_size = max(bucket_size, batch_size)
        self.seed = seed
        self.drop_last = drop_last
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.epoch = 0
        self.progress = 0.0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self.progress = 0.0

    def _shuffled_indices(self, rng: random.Random):
        if not self.shuffle:
            yield from range(len(self.dataset))
            return
        source = iter(range(len(self.dataset)))
        buffer = []
        for _ in range(self.bucket_size):
            try:
                buffer.append(next(source))
            except StopIteration:
                break
        for index in source:
            position = rng.randrange(len(buffer))
            yield buffer[position]
            buffer[position] = index
        rng.shuffle(buffer)
        yield from buffer

    def _sample_length(self, index: int) -> SampleLength:
        return _compute_sample_length(self.dataset[index], self.audio_vae_fps, self.patch_size)

    @staticmethod
    def _cost(lengths: List[SampleLength]) -> int:
        count = len(lengths)
        return count * (
            max(length.text_tokens for length in lengths)
            + max(length.target_sequence for length in lengths)
            + max(length.ref_sequence for length in lengths)
            + (4 if lengths[0].has_ref_audio else 2)
        )

    def _window_batches(self, indices: List[int]) -> List[List[int]]:
        measured = [(index, self._sample_length(index)) for index in indices]
        measured = [(index, length) for index, length in measured if length.packed <= self.max_batch_tokens]
        batches = []
        for has_ref in (False, True):
            group = [(index, length) for index, length in measured if length.has_ref_audio == has_ref]
            group.sort(key=lambda item: (item[1].target_samples, item[1].ref_samples, item[1].packed))
            current_indices: List[int] = []
            current_lengths: List[SampleLength] = []
            for index, length in group:
                prospective_lengths = [*current_lengths, length]
                if current_indices and (
                    len(prospective_lengths) > self.batch_size
                    or self._cost(prospective_lengths) > self.max_batch_tokens
                ):
                    batches.append(current_indices)
                    current_indices, current_lengths = [], []
                current_indices.append(index)
                current_lengths.append(length)
            if current_indices:
                batches.append(current_indices)
        return batches

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        stream = self._shuffled_indices(rng)
        processed = 0
        pending_batches: List[List[int]] = []
        while True:
            window = []
            try:
                for _ in range(self.bucket_size):
                    window.append(next(stream))
            except StopIteration:
                pass
            if not window:
                break
            batches = self._window_batches(window)
            if self.shuffle:
                rng.shuffle(batches)
            pending_batches.extend(batches)
            processed += len(window)
            self.progress = processed / max(1, len(self.dataset))
            complete_count = len(pending_batches) // self.world_size * self.world_size
            complete_batches = pending_batches[:complete_count]
            pending_batches = pending_batches[complete_count:]
            yield from complete_batches[self.rank :: self.world_size]

        if pending_batches and not self.drop_last:
            original_count = len(pending_batches)
            pending_batches.extend(
                [list(pending_batches[i % original_count]) for i in range(self.world_size - original_count)]
            )
            yield from pending_batches[self.rank :: self.world_size]

    def __len__(self) -> int:
        # Dynamic token limits make the exact value unknowable without scanning.
        return math.ceil(len(self.dataset) / max(1, self.batch_size * self.world_size))


def select_prompt_audio(
    candidates: List[Dict],
    strategy: str,
    *,
    seed: int = 42,
    index: int = 0,
    base_dir: Optional[Path] = None,
) -> Optional[str]:
    """Select one reference path from a manifest ``prompt_audio_list``."""
    if not candidates:
        return None
    if not isinstance(candidates, list) or any(not isinstance(candidate, dict) for candidate in candidates):
        raise ValueError("prompt_audio_list must be a list of objects")
    if any(not isinstance(candidate.get("audio"), str) or not candidate["audio"] for candidate in candidates):
        raise ValueError("Each prompt_audio_list item must contain a non-empty 'audio' path")

    if strategy == "random":
        selected = random.Random(seed + index).choice(candidates)
    elif strategy == "highest_ssim":
        try:
            selected = max(candidates, key=lambda candidate: float(candidate["ssim"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("highest_ssim selection requires a numeric 'ssim' in every prompt audio item") from exc
    elif strategy == "shortest":
        import soundfile as sf

        try:
            selected = min(
                candidates,
                key=lambda candidate: sf.info(
                    str(base_dir / candidate["audio"])
                    if base_dir is not None and not Path(candidate["audio"]).is_absolute()
                    else candidate["audio"]
                ).duration,
            )
        except Exception as exc:
            raise ValueError(f"Cannot determine prompt audio duration: {exc}") from exc
    else:
        raise ValueError(f"Unknown ref_audio_selection '{strategy}'; expected one of: shortest, random, highest_ssim")
    selected_path = Path(selected["audio"])
    if base_dir is not None and not selected_path.is_absolute():
        selected_path = base_dir / selected_path
    return str(selected_path)


@argbind.bind()
def load_audio_text_datasets(
    train_manifest: str,
    val_manifest: str = "",
    text_column: str = DEFAULT_TEXT_COLUMN,
    audio_column: str = DEFAULT_AUDIO_COLUMN,
    prompt_audio_list_column: str = DEFAULT_PROMPT_AUDIO_LIST_COLUMN,
    ref_audio_selection: str = "highest_ssim",
    ref_audio_seed: int = 42,
    dataset_id_column: str = DEFAULT_ID_COLUMN,
    sample_rate: int = 16_000,
    num_proc: int = 1,
) -> Tuple[Dataset, Optional[Dataset]]:
    valid_selection_strategies = {"shortest", "random", "highest_ssim"}
    if ref_audio_selection not in valid_selection_strategies:
        raise ValueError(
            f"Unknown ref_audio_selection '{ref_audio_selection}'; "
            f"expected one of: {', '.join(sorted(valid_selection_strategies))}"
        )

    data_files = {"train": train_manifest}
    if val_manifest:
        data_files["validation"] = val_manifest

    dataset_dict: DatasetDict = load_dataset("json", data_files=data_files)

    def prepare(ds: Dataset, manifest_path: str) -> Dataset:
        if audio_column not in ds.column_names:
            raise ValueError(f"Expected '{audio_column}' column in manifest.")
        if audio_column != DEFAULT_AUDIO_COLUMN:
            ds = ds.rename_column(audio_column, DEFAULT_AUDIO_COLUMN)
        if text_column != DEFAULT_TEXT_COLUMN:
            ds = ds.rename_column(text_column, DEFAULT_TEXT_COLUMN)

        if DEFAULT_REF_AUDIO_COLUMN in ds.column_names:
            raise ValueError(
                "The legacy 'ref_audio' manifest column is no longer supported; "
                f"use '{prompt_audio_list_column}' instead"
            )

        # Select one candidate once during dataset preparation. Random selection
        # is deterministic for a given seed and row index across all DDP ranks.
        if prompt_audio_list_column in ds.column_names:
            selected_column = "_selected_ref_audio"

            def select_reference(item, index):
                return {
                    selected_column: select_prompt_audio(
                        item[prompt_audio_list_column],
                        ref_audio_selection,
                        seed=ref_audio_seed,
                        index=index,
                        base_dir=Path(manifest_path).resolve().parent,
                    )
                }

            ds = ds.map(select_reference, with_indices=True)
            ds = ds.remove_columns(prompt_audio_list_column)
            ds = ds.rename_column(selected_column, DEFAULT_REF_AUDIO_COLUMN)
            ds = ds.cast_column(DEFAULT_REF_AUDIO_COLUMN, Audio(sampling_rate=sample_rate))

        ds = ds.cast_column(DEFAULT_AUDIO_COLUMN, Audio(sampling_rate=sample_rate))

        if dataset_id_column and dataset_id_column in ds.column_names:
            if dataset_id_column != DEFAULT_ID_COLUMN:
                ds = ds.rename_column(dataset_id_column, DEFAULT_ID_COLUMN)
        else:
            ds = ds.add_column(DEFAULT_ID_COLUMN, [0] * len(ds))
        return ds

    train_ds = prepare(dataset_dict["train"], train_manifest)
    val_ds = prepare(dataset_dict["validation"], val_manifest) if "validation" in dataset_dict else None
    return train_ds, val_ds


def compute_sample_lengths(
    ds: Dataset,
    audio_vae_fps: float = 25,
    patch_size: int = 1,
) -> List[SampleLength]:
    """
    预估每个样本经过 packer 之后的大致序列长度（text+audio），用于过滤超长样本。

    逻辑与 AudioFeatureProcessingPacker / AudioVAE 一致：
    - 文本长度: len(text_ids)
    - 音频长度:
        duration(s) * audio_vae_fps -> 近似 VAE 帧数 t_vae
        t_seq = ceil(t_vae / patch_size)
    返回的 SampleLength 包括：
    - 文本 token 数
    - target audio 原始采样点数及预估 patch 序列长度
    - ref audio 原始采样点数及预估 patch 序列长度（无参考音频时均为 0）
    - packed 总长度：无 ref 为 text_len + t_seq + 2，有 ref 为
      text_len + t_seq + ref_seq + 4

    Raw sample counts are retained so batching can account for target/ref audio
    padding independently rather than relying only on their packed total.
    """
    lengths = []
    for item in ds:
        lengths.append(_compute_sample_length(item, audio_vae_fps, patch_size))

    return lengths


class HFVoxCPMDataset(TorchDataset):
    """
    Thin wrapper around a tokenized HuggingFace dataset that returns
    PyTorch-friendly samples.
    """

    _SENTINEL = [-100.0]

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.has_ref_audio = DEFAULT_REF_AUDIO_COLUMN in dataset.column_names

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        item = self.dataset[idx]
        audio = item[DEFAULT_AUDIO_COLUMN]
        sample = {
            "text_ids": item["text_ids"],
            "audio_array": audio["array"],
            "audio_sampling_rate": audio["sampling_rate"],
            "dataset_id": item.get(DEFAULT_ID_COLUMN, 0),
            "is_prompt": item.get("is_prompt", False),
        }
        for column in METADATA_COLUMNS:
            if column in self.dataset.column_names:
                sample[column] = item.get(column)
        if self.has_ref_audio:
            ref = item.get(DEFAULT_REF_AUDIO_COLUMN)
            sample["ref_audio_array"] = ref["array"] if ref else self._SENTINEL
        return sample

    @staticmethod
    def pad_sequences(seqs: List[torch.Tensor], pad_value: float):
        if not seqs:
            return torch.empty(0)
        max_len = max(seq.shape[0] for seq in seqs)
        padded = []
        for seq in seqs:
            if seq.shape[0] < max_len:
                pad_width = (0, max_len - seq.shape[0])
                seq = torch.nn.functional.pad(seq, pad_width, value=pad_value)
            padded.append(seq)
        return torch.stack(padded)

    @classmethod
    def collate_fn(cls, batch: List[Dict]):
        text_tensors = [torch.tensor(sample["text_ids"], dtype=torch.int32) for sample in batch]
        audio_tensors = [torch.tensor(sample["audio_array"], dtype=torch.float32) for sample in batch]
        dataset_ids = torch.tensor([sample["dataset_id"] for sample in batch], dtype=torch.int32)
        is_prompts = [bool(sample.get("is_prompt", False)) for sample in batch]

        text_padded = cls.pad_sequences(text_tensors, pad_value=-100)
        audio_padded = cls.pad_sequences(audio_tensors, pad_value=-100.0)
        task_ids = torch.ones(text_padded.size(0), dtype=torch.int32)

        result = {
            "text_tokens": text_padded,
            "audio_tokens": audio_padded,
            "task_ids": task_ids,
            "dataset_ids": dataset_ids,
            "is_prompts": is_prompts,
        }
        for column in METADATA_COLUMNS:
            if column in batch[0]:
                result[column] = [sample.get(column) for sample in batch]

        if "ref_audio_array" in batch[0]:
            ref_tensors = [torch.tensor(s["ref_audio_array"], dtype=torch.float32) for s in batch]
            result["ref_audio_tokens"] = cls.pad_sequences(ref_tensors, pad_value=-100.0)

        return result


class BatchProcessor:
    """
    Wraps ``AudioFeatureProcessingPacker`` so the training loop can mirror
    the minicpm-audio mechanics.
    """

    def __init__(
        self,
        *,
        config: VoxCPMConfig,
        audio_vae: AudioVAE,
        dataset_cnt: int,
        device: torch.device,
    ):
        self.device = device
        self.dataset_cnt = dataset_cnt
        self.audio_vae = audio_vae
        self.audio_vae.to(device)
        self.packer = AudioFeatureProcessingPacker(
            dataset_cnt=dataset_cnt,
            max_len=config.max_length,
            patch_size=config.patch_size,
            feat_dim=config.feat_dim,
            audio_vae=self.audio_vae,
        )

    def __call__(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        audio_tokens = batch["audio_tokens"].to(self.device)
        text_tokens = batch["text_tokens"].to(self.device)
        task_ids = batch["task_ids"].to(self.device)
        dataset_ids = batch["dataset_ids"].to(self.device)

        ref_audio_tokens = None
        if "ref_audio_tokens" in batch:
            ref_audio_tokens = batch["ref_audio_tokens"].to(self.device)

        packed = self.packer(
            audio_tokens=audio_tokens,
            text_tokens=text_tokens,
            task_ids=task_ids,
            dataset_ids=dataset_ids,
            is_prompts=batch["is_prompts"],
            ref_audio_tokens=ref_audio_tokens,
        )
        return packed


def build_dataloader(
    hf_dataset: Dataset,
    *,
    accelerator,
    batch_size: int,
    num_workers: int,
    drop_last: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int = 0,
    worker_cpu_threads: int = 0,
    shuffle: bool = True,
    sample_lengths: Optional[List[SampleLength]] = None,
    max_batch_tokens: int = 0,
    seed: int = 42,
    length_bucket_size: int = 0,
    audio_vae_fps: float = 25,
    patch_size: int = 1,
) -> torch.utils.data.DataLoader:
    torch_dataset = HFVoxCPMDataset(hf_dataset)
    batch_sampler = None
    if max_batch_tokens > 0:
        if sample_lengths is not None:
            if len(sample_lengths) != len(torch_dataset):
                raise ValueError("sample_lengths must match the dataset when dynamic batching is enabled")
            batch_sampler = DynamicBatchSampler(
                sample_lengths,
                max_batch_tokens=max_batch_tokens,
                batch_size=batch_size,
                seed=seed,
                drop_last=drop_last,
                rank=accelerator.rank,
                world_size=accelerator.world_size,
                shuffle=shuffle,
                bucket_size=length_bucket_size,
            )
        else:
            batch_sampler = BufferedDynamicBatchSampler(
                hf_dataset,
                audio_vae_fps=audio_vae_fps,
                patch_size=patch_size,
                max_batch_tokens=max_batch_tokens,
                batch_size=batch_size,
                bucket_size=length_bucket_size,
                seed=seed,
                drop_last=drop_last,
                rank=accelerator.rank,
                world_size=accelerator.world_size,
                shuffle=shuffle,
            )

    return accelerator.prepare_dataloader(
        torch_dataset,
        batch_size=batch_size,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        shuffle=shuffle,
        collate_fn=HFVoxCPMDataset.collate_fn,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        worker_cpu_threads=worker_cpu_threads,
    )
