#!/usr/bin/env python3

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import contextlib
from typing import Dict

import argbind
import torch
from tensorboardX import SummaryWriter
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
import signal
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    from safetensors.torch import save_file

    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    print("Warning: safetensors not available, will use pytorch format", file=sys.stderr)

import json

from voxcpm.model import VoxCPMModel, VoxCPM2Model
from voxcpm.model.voxcpm import LoRAConfig as LoRAConfigV1
from voxcpm.model.voxcpm2 import LoRAConfig as LoRAConfigV2
from voxcpm.training import (
    Accelerator,
    BatchProcessor,
    TrainingTracker,
    build_dataloader,
    load_audio_text_datasets,
)


@argbind.bind(without_prefix=True)
def train(
    pretrained_path: str,
    train_manifest: str,
    val_manifest: str = "",
    sample_rate: int = 16_000,
    out_sample_rate: int = 0,  # AudioVAE decoder output rate; used for TensorBoard audio logging
    batch_size: int = 1,
    grad_accum_steps: int = 1,
    num_workers: int = 2,
    persistent_workers: bool = False,
    prefetch_factor: int = 0,
    worker_cpu_threads: int = 0,
    num_iters: int = 100_000,
    log_interval: int = 100,
    valid_interval: int = 1_000,
    quick_valid_batches: int = 10,
    full_valid_interval: int = 0,
    val_audio_interval: int = 1_000,
    val_audio_num_samples: int = 2,
    val_audio_per_dialect: int = 1,
    save_val_audio_wavs: bool = False,
    val_audio_dir: str = "",
    save_interval: int = 10_000,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-2,
    warmup_steps: int = 1_000,
    max_steps: int = 100_000,
    max_batch_tokens: int = 0,
    save_path: str = "checkpoints",
    tensorboard: str = "",
    lambdas: Dict[str, float] = {"loss/diff": 1.0, "loss/stop": 1.0},
    lora: dict = None,
    config_path: str = "",
    max_grad_norm: float = 0.0,  # gradient clipping; 0 = disabled (backward compat)
    # Distribution options (for LoRA checkpoints)
    hf_model_id: str = "",  # HuggingFace model ID (e.g., "openbmb/VoxCPM1.5")
    distribute: bool = False,  # If True, save hf_model_id as base_model; otherwise save pretrained_path
):
    _ = config_path

    # Validate distribution options
    if lora is not None and distribute and not hf_model_id:
        raise ValueError("hf_model_id is required when distribute=True")

    accelerator = Accelerator(amp=True)

    save_dir = Path(save_path)
    tb_dir = Path(tensorboard) if tensorboard else save_dir / "logs"

    # Only create directories on rank 0 to avoid race conditions
    if accelerator.rank == 0:
        save_dir.mkdir(parents=True, exist_ok=True)
        tb_dir.mkdir(parents=True, exist_ok=True)
    accelerator.barrier()  # Wait for directory creation

    writer = SummaryWriter(log_dir=str(tb_dir)) if accelerator.rank == 0 else None
    tracker = TrainingTracker(writer=writer, log_file=str(save_dir / "train.log"), rank=accelerator.rank)

    # Auto-detect model architecture from config.json
    with open(os.path.join(pretrained_path, "config.json"), "r", encoding="utf-8") as _f:
        _arch = json.load(_f).get("architecture", "voxcpm").lower()
    _model_cls = VoxCPM2Model if _arch == "voxcpm2" else VoxCPMModel
    LoRAConfig = LoRAConfigV2 if _arch == "voxcpm2" else LoRAConfigV1
    if accelerator.rank == 0:
        print(f"Detected architecture: {_arch} -> {_model_cls.__name__}", file=sys.stderr)
    base_model = _model_cls.from_local(
        pretrained_path, optimize=False, training=True, lora_config=LoRAConfig(**lora) if lora else None
    )
    tokenizer = base_model.text_tokenizer

    expected_sr = base_model.audio_vae.sample_rate
    assert sample_rate == expected_sr, (
        f"sample_rate mismatch: config says {sample_rate}, but the AudioVAE encoder expects {expected_sr}. "
        f"Please set sample_rate: {expected_sr} in your training config. "
    )

    train_ds, val_ds = load_audio_text_datasets(
        train_manifest=train_manifest,
        val_manifest=val_manifest,
        sample_rate=sample_rate,
    )

    def tokenize(batch):
        text_list = batch["text"]
        text_ids = [tokenizer(text) for text in text_list]
        return {"text_ids": text_ids}

    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"])
    # Save original validation texts for audio generation display
    val_texts = None
    val_dialects = []
    if val_ds is not None:
        if "dialect" in val_ds.column_names:
            val_dialects = sorted(str(item) for item in set(val_ds["dialect"]))
        val_texts = list(val_ds["text"])  # Save original texts
        val_ds = val_ds.map(tokenize, batched=True, remove_columns=["text"])

    dataset_cnt = int(max(train_ds["dataset_id"])) + 1 if "dataset_id" in train_ds.column_names else 1
    num_train_samples = len(train_ds)

    # ------------------------------------------------------------------ #
    # Optional: filter samples by estimated token count to avoid OOM
    # Enabled when max_batch_tokens > 0:
    #   max_sample_len = max_batch_tokens // batch_size
    #   Samples exceeding this length will be dropped
    # ------------------------------------------------------------------ #
    if max_batch_tokens and max_batch_tokens > 0:
        from voxcpm.training.data import compute_sample_lengths

        audio_vae_fps = base_model.audio_vae.sample_rate / base_model.audio_vae.hop_length
        est_lengths = compute_sample_lengths(
            train_ds,
            audio_vae_fps=audio_vae_fps,
            patch_size=base_model.config.patch_size,
        )
        max_sample_len = max_batch_tokens // batch_size if batch_size > 0 else max(est_lengths)
        keep_indices = [i for i, L in enumerate(est_lengths) if L <= max_sample_len]

        if len(keep_indices) < len(train_ds) and accelerator.rank == 0:
            tracker.print(
                f"Filtering {len(train_ds) - len(keep_indices)} / {len(train_ds)} "
                f"training samples longer than {max_sample_len} tokens "
                f"(max_batch_tokens={max_batch_tokens})."
            )
        train_ds = train_ds.select(keep_indices)

    train_loader = build_dataloader(
        train_ds,
        accelerator=accelerator,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        worker_cpu_threads=worker_cpu_threads,
    )
    val_loader = (
        build_dataloader(
            val_ds,
            accelerator=accelerator,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=False,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            worker_cpu_threads=worker_cpu_threads,
            shuffle=False,
        )
        if val_ds is not None
        else None
    )
    full_val_loader = (
        build_dataloader(
            val_ds,
            accelerator=accelerator,
            batch_size=1,
            num_workers=num_workers,
            drop_last=False,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            worker_cpu_threads=worker_cpu_threads,
            shuffle=False,
        )
        if val_ds is not None and full_valid_interval and full_valid_interval > 0
        else None
    )

    batch_processor = BatchProcessor(
        config=base_model.config,
        audio_vae=base_model.audio_vae,
        dataset_cnt=dataset_cnt,
        device=accelerator.device,
    )
    # Save audio_vae and output sample rate for audio generation.
    # Prefer model's actual output rate; fall back to YAML out_sample_rate or encode rate.
    audio_vae_for_gen = base_model.audio_vae
    out_sr = base_model.sample_rate  # decoder output rate (e.g. 48000 for V2)
    if out_sr == 0 and out_sample_rate > 0:
        out_sr = out_sample_rate
    del base_model.audio_vae
    model = accelerator.prepare_model(base_model)
    unwrapped_model = accelerator.unwrap(model)
    unwrapped_model.train()

    # Only print param info on rank 0 to avoid cluttered output
    if accelerator.rank == 0:
        for name, param in model.named_parameters():
            print(name, param.requires_grad, file=sys.stderr)

    optimizer = AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # Cosine + warmup scheduler from transformers:
    # - num_warmup_steps: warmup steps
    # - num_training_steps: total training steps (outer step count)
    total_training_steps = max_steps if max_steps > 0 else num_iters
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    # All ranks load the same checkpoint to keep model and optimizer state in sync.
    start_step = load_checkpoint(model, optimizer, scheduler, save_dir, rank=accelerator.rank)
    accelerator.barrier()

    if start_step > 0 and accelerator.rank == 0:
        tracker.print(f"Resuming training from step {start_step}")

    # Resume tracker for signal handler to read current step
    resume = {"step": start_step}

    # Register signal handler to save checkpoint on termination (SIGTERM/SIGINT)
    def _signal_handler(
        signum,
        frame,
        _model=model,
        _optim=optimizer,
        _sched=scheduler,
        _save_dir=save_dir,
        _pretrained=pretrained_path,
        _hf_id=hf_model_id,
        _dist=distribute,
        _resume=resume,
        _rank=accelerator.rank,
    ):
        try:
            cur_step = int(_resume.get("step", start_step))
        except Exception:
            cur_step = start_step
        if _rank == 0:
            print(f"Signal {signum} received. Saving checkpoint at step {cur_step} ...", file=sys.stderr)
            try:
                save_checkpoint(_model, _optim, _sched, _save_dir, cur_step, _pretrained, _hf_id, _dist)
                print("Checkpoint saved. Exiting.", file=sys.stderr)
            except Exception as e:
                print(f"Error saving checkpoint on signal: {e}", file=sys.stderr)
        os._exit(0)

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    # Manual epoch management instead of itertools.cycle to support DistributedSampler.set_epoch()
    grad_accum_steps = max(int(grad_accum_steps), 1)
    data_epoch = 0
    train_iter = iter(train_loader)

    def get_next_batch():
        """Get next batch, handles epoch boundary and DistributedSampler."""
        nonlocal train_iter, data_epoch
        try:
            return next(train_iter)
        except StopIteration:
            data_epoch += 1
            # Key: set DistributedSampler epoch to ensure different data order each epoch
            sampler = getattr(train_loader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(data_epoch)
            train_iter = iter(train_loader)
            return next(train_iter)

    with tracker.live():
        for step in range(start_step, num_iters):
            # update resume step so signal handler can save current progress
            resume["step"] = step
            tracker.step = step
            optimizer.zero_grad(set_to_none=True)

            # Gradient accumulation: accumulate gradients over micro-batches before optimizer step
            loss_dict = {}
            for micro_step in range(grad_accum_steps):
                batch = get_next_batch()
                processed = batch_processor(batch)

                # Only sync gradients on the last micro-batch
                # Use no_sync() for intermediate steps to reduce communication overhead
                is_last_micro_step = micro_step == grad_accum_steps - 1
                sync_context = contextlib.nullcontext() if is_last_micro_step else accelerator.no_sync()

                with sync_context:
                    with accelerator.autocast(dtype=torch.bfloat16):
                        outputs = model(
                            processed["text_tokens"],
                            processed["text_mask"],
                            processed["audio_feats"],
                            processed["audio_mask"],
                            processed["loss_mask"],
                            processed["position_ids"],
                            processed["labels"],
                            progress=step / max(1, num_iters),
                        )

                    total_loss = 0.0
                    for key, value in outputs.items():
                        if key.startswith("loss/"):
                            weight = lambdas.get(key, 1.0)
                            loss_value = value * weight / grad_accum_steps
                            total_loss = total_loss + loss_value
                            # Record raw loss from last micro-batch for logging
                            loss_dict[key] = value.detach()

                    # Accumulate gradients (normalized by grad_accum_steps)
                    accelerator.backward(total_loss)

            # After all micro-batches, do unscale / grad_norm / step
            scaler = getattr(accelerator, "scaler", None)
            if scaler is not None:
                scaler.unscale_(optimizer)
            effective_max_norm = max_grad_norm if max_grad_norm > 0 else 1e9
            grad_norm = torch.nn.utils.clip_grad_norm_(unwrapped_model.parameters(), max_norm=effective_max_norm)

            accelerator.step(optimizer)
            accelerator.update()
            scheduler.step()

            if step % log_interval == 0 or step == num_iters - 1:
                loss_values = {k: v.item() if isinstance(v, torch.Tensor) else float(v) for k, v in loss_dict.items()}
                loss_values["lr"] = float(optimizer.param_groups[0]["lr"])
                # Account for all GPUs when converting steps to epochs.
                epoch = (step * grad_accum_steps * batch_size * accelerator.world_size) / max(1, num_train_samples)
                loss_values["epoch"] = float(epoch)
                loss_values["grad_norm"] = float(grad_norm)
                tracker.log_metrics(loss_values, split="train")

            is_last_step = step == num_iters - 1
            if val_loader is not None and (step % valid_interval == 0 or is_last_step):
                validate_loss(
                    model,
                    val_loader,
                    batch_processor,
                    accelerator,
                    tracker,
                    lambdas,
                    step=step,
                    split="val/quick",
                    max_batches=quick_valid_batches,
                    group_by="",
                )

            if (
                full_val_loader is not None
                and full_valid_interval
                and full_valid_interval > 0
                and (step % full_valid_interval == 0 or is_last_step)
            ):
                validate_loss(
                    model,
                    full_val_loader,
                    batch_processor,
                    accelerator,
                    tracker,
                    lambdas,
                    step=step,
                    split="val/full",
                    max_batches=0,
                    group_by="dialect",
                    group_values=val_dialects,
                )

            if (
                val_loader is not None
                and val_audio_interval
                and val_audio_interval > 0
                and (step % val_audio_interval == 0 or is_last_step)
            ):
                validate_audio(
                    model,
                    val_ds,
                    audio_vae_for_gen,
                    writer,
                    step,
                    accelerator,
                    sample_rate=sample_rate,
                    out_sample_rate=out_sr,
                    val_texts=val_texts,
                    num_samples=val_audio_num_samples,
                    samples_per_dialect=val_audio_per_dialect,
                    save_wavs=save_val_audio_wavs,
                    wav_dir=val_audio_dir or str(save_dir / "val_audio"),
                    tracker=tracker,
                )

            if (step % save_interval == 0 or step == num_iters - 1) and accelerator.rank == 0:
                save_checkpoint(model, optimizer, scheduler, save_dir, step, pretrained_path, hf_model_id, distribute)

    if accelerator.rank == 0:
        save_checkpoint(model, optimizer, scheduler, save_dir, num_iters, pretrained_path, hf_model_id, distribute)
    if writer:
        writer.close()


def _metric_value(value):
    return value.item() if isinstance(value, torch.Tensor) else float(value)


def _all_reduce_sum(accelerator, value: torch.Tensor):
    if hasattr(accelerator, "all_reduce"):
        return accelerator.all_reduce(value, op=torch.distributed.ReduceOp.SUM)
    return value


def validate_loss(
    model,
    val_loader,
    batch_processor,
    accelerator,
    tracker,
    lambdas,
    step=0,
    split="val/quick",
    max_batches=10,
    group_by="",
    group_values=None,
):
    """Run validation loss only.

    ``max_batches`` > 0 performs quick validation on the first N validation
    batches. ``max_batches`` <= 0 runs the full validation loader. Losses are
    weighted by the number of valid target audio tokens in ``loss_mask``.
    """
    from collections import defaultdict

    model.eval()
    device = accelerator.device
    loss_sums = defaultdict(lambda: torch.zeros((), device=device, dtype=torch.float32))
    group_loss_sums = defaultdict(lambda: defaultdict(lambda: torch.zeros((), device=device, dtype=torch.float32)))
    group_token_sums = defaultdict(lambda: torch.zeros((), device=device, dtype=torch.float32))
    token_sum = torch.zeros((), device=device, dtype=torch.float32)
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            if max_batches and max_batches > 0 and num_batches >= max_batches:
                break

            processed = batch_processor(batch)
            valid_tokens = processed["loss_mask"].sum().detach().to(device=device, dtype=torch.float32)
            valid_tokens = torch.clamp(valid_tokens, min=1.0)

            with accelerator.autocast(dtype=torch.bfloat16):
                outputs = model(
                    processed["text_tokens"],
                    processed["text_mask"],
                    processed["audio_feats"],
                    processed["audio_mask"],
                    processed["loss_mask"],
                    processed["position_ids"],
                    processed["labels"],
                    progress=0.0,
                    sample_generate=False,
                )

            total = torch.zeros((), device=device, dtype=torch.float32)
            batch_loss_values = {}
            for key, value in outputs.items():
                if key.startswith("loss/"):
                    detached = value.detach().to(device=device, dtype=torch.float32)
                    batch_loss_values[key] = detached
                    loss_sums[key] += detached * valid_tokens
                    total = total + lambdas.get(key, 1.0) * detached
            loss_sums["loss/total"] += total * valid_tokens
            token_sum += valid_tokens

            if group_by and group_by in batch and batch[group_by]:
                groups = [str(group) if group is not None else "unknown" for group in batch[group_by]]
                sample_tokens = processed["loss_mask"].sum(dim=1).detach().to(device=device, dtype=torch.float32)
                for group in sorted(set(groups)):
                    mask = torch.tensor([item == group for item in groups], device=device, dtype=torch.bool)
                    group_tokens = torch.clamp(sample_tokens[mask].sum(), min=1.0)
                    group_token_sums[group] += group_tokens
                    for key, value in batch_loss_values.items():
                        group_loss_sums[group][key] += value * group_tokens
                    group_loss_sums[group]["loss/total"] += total * group_tokens

            num_batches += 1

    if num_batches == 0:
        if accelerator.rank == 0:
            tracker.print(f"[Warning] Skip {split}: no validation batches were processed")
        model.train()
        return

    token_sum = _all_reduce_sum(accelerator, token_sum)
    metrics = {
        "tokens": _metric_value(token_sum),
        "batches": float(num_batches),
    }
    for key, value in loss_sums.items():
        reduced = _all_reduce_sum(accelerator, value)
        metrics[key] = _metric_value(reduced / torch.clamp(token_sum, min=1.0))

    if group_by and group_values:
        for group in group_values:
            reduced_group_tokens = _all_reduce_sum(accelerator, group_token_sums[group])
            for key in loss_sums:
                value = group_loss_sums[group][key]
                reduced_value = _all_reduce_sum(accelerator, value)
                metric_name = f"{group_by}/{group}/{key}"
                metrics[metric_name] = _metric_value(reduced_value / torch.clamp(reduced_group_tokens, min=1.0))

    tracker.log_metrics(metrics, split=split)
    model.train()


def _audio_array(sample, column, sample_rate):
    import numpy as np

    audio = sample.get(column)
    if not isinstance(audio, dict) or "array" not in audio:
        return None, None, None

    audio_np = np.array(audio["array"], dtype=np.float32)
    audio_sr = audio.get("sampling_rate", sample_rate)
    audio_path = audio.get("path")
    return audio_np, audio_sr, audio_path


def _resample_audio(audio_np, source_sr, target_sr):
    if audio_np is None or source_sr == target_sr:
        return audio_np
    import torchaudio.functional as F

    return F.resample(torch.from_numpy(audio_np).unsqueeze(0), source_sr, target_sr).squeeze(0).numpy()


def _safe_tag(value):
    text = str(value) if value is not None else "unknown"
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)


def _select_audio_validation_indices(val_ds, num_samples: int, samples_per_dialect: int):
    if len(val_ds) == 0 or num_samples <= 0:
        return []

    if "dialect" not in val_ds.column_names:
        return list(range(min(num_samples, len(val_ds))))

    selected = []
    per_dialect_counts = {}
    for idx in range(len(val_ds)):
        sample = val_ds[idx]
        dialect = str(sample.get("dialect") or "unknown")
        count = per_dialect_counts.get(dialect, 0)
        if count >= max(samples_per_dialect, 1):
            continue
        selected.append(idx)
        per_dialect_counts[dialect] = count + 1
        if len(selected) >= num_samples:
            break

    if not selected:
        return list(range(min(num_samples, len(val_ds))))
    return selected


def validate_audio(
    model,
    val_ds,
    audio_vae,
    writer,
    step,
    accelerator,
    sample_rate=22050,
    out_sample_rate=0,
    val_texts=None,
    num_samples=2,
    samples_per_dialect=1,
    save_wavs=False,
    wav_dir="",
    tracker=None,
):
    """Generate fixed validation samples, log them to TensorBoard, and optionally save wavs."""
    if writer is None or val_ds is None or audio_vae is None or accelerator.rank != 0:
        return

    import numpy as np

    log = tracker.print if tracker else print
    selected_indices = _select_audio_validation_indices(val_ds, num_samples, samples_per_dialect)
    log(f"[Audio] Starting audio validation for {len(selected_indices)} samples at step {step}")

    unwrapped_model = accelerator.unwrap(model)
    gen_sr = out_sample_rate if out_sample_rate > 0 else sample_rate
    wav_root = Path(wav_dir) / f"step_{step:07d}" if save_wavs and wav_dir else None
    if wav_root is not None:
        wav_root.mkdir(parents=True, exist_ok=True)

    for sample_idx in selected_indices:
        sample = val_ds[sample_idx]
        text = val_texts[sample_idx] if val_texts and sample_idx < len(val_texts) else "Hello, this is a test."
        sample_id = sample.get("sample_id", f"sample_{sample_idx}")
        dialect = sample.get("dialect", "unknown")
        tag = f"val_audio/{_safe_tag(dialect)}/{_safe_tag(sample_id)}"

        gt_audio_np = None
        ref_audio_np = None
        reference_wav_path = ""
        try:
            gt_audio_np, gt_sr, _ = _audio_array(sample, "audio", sample_rate)
            gt_audio_np = _resample_audio(gt_audio_np, gt_sr, sample_rate) if gt_audio_np is not None else None

            ref_audio_np, ref_sr, ref_path = _audio_array(sample, "ref_audio", sample_rate)
            ref_audio_np = _resample_audio(ref_audio_np, ref_sr, sample_rate) if ref_audio_np is not None else None
            reference_wav_path = ref_path or ""
        except Exception as e:
            log(f"[Warning] Failed to load validation audio for sample {sample_idx}: {e}")

        prev_training = unwrapped_model.training
        try:
            unwrapped_model.eval()
            unwrapped_model.audio_vae = audio_vae.to(torch.float32)

            generate_kwargs = {
                "target_text": text,
                "inference_timesteps": 10,
                "cfg_value": 2.0,
                "seed": 42,
            }
            if reference_wav_path:
                generate_kwargs["reference_wav_path"] = reference_wav_path
                log(f"[Audio] Generating sample {sample_idx} with ref_audio: {reference_wav_path}")
            else:
                log(f"[Audio] Generating sample {sample_idx} without ref_audio")

            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if torch.cuda.is_available()
                else contextlib.nullcontext()
            )
            with torch.no_grad():
                with autocast_ctx:
                    generated = unwrapped_model.generate(**generate_kwargs)

            if generated is None or len(generated) == 0:
                log(f"[Warning] Generated audio is empty for sample {sample_idx}")
                continue

            gen_audio_np = (
                generated.cpu().float().numpy().flatten()
                if isinstance(generated, torch.Tensor)
                else np.array(generated, dtype=np.float32).flatten()
            )
            gen_audio_np = normalize_audio(gen_audio_np)

            writer.add_audio(f"{tag}/generated", gen_audio_np, global_step=step, sample_rate=gen_sr)
            if gt_audio_np is not None:
                writer.add_audio(
                    f"{tag}/ground_truth", normalize_audio(gt_audio_np), global_step=step, sample_rate=sample_rate
                )
            if ref_audio_np is not None:
                writer.add_audio(
                    f"{tag}/conditioning_ref", normalize_audio(ref_audio_np), global_step=step, sample_rate=sample_rate
                )

            if wav_root is not None:
                import soundfile as sf

                sf.write(wav_root / f"{_safe_tag(dialect)}_{_safe_tag(sample_id)}_generated.wav", gen_audio_np, gen_sr)
                if gt_audio_np is not None:
                    sf.write(
                        wav_root / f"{_safe_tag(dialect)}_{_safe_tag(sample_id)}_ground_truth.wav",
                        normalize_audio(gt_audio_np),
                        sample_rate,
                    )
                if ref_audio_np is not None:
                    sf.write(
                        wav_root / f"{_safe_tag(dialect)}_{_safe_tag(sample_id)}_conditioning_ref.wav",
                        normalize_audio(ref_audio_np),
                        sample_rate,
                    )

            try:
                mel_gen = compute_mel_spectrogram(gen_audio_np, gen_sr)
                mel_gt = compute_mel_spectrogram(gt_audio_np, sample_rate) if gt_audio_np is not None else None
                fig = create_mel_figure(gen_audio_np, mel_gen, gen_sr, step, gt_audio_np, mel_gt)
                writer.add_figure(f"{tag}/mel_spectrogram", fig, global_step=step)
            except Exception as e:
                log(f"[Warning] Failed to create mel spectrogram for sample {sample_idx}: {e}")

        except Exception as e:
            log(f"[Warning] Failed to generate validation audio for sample {sample_idx}: {e}")
            import traceback

            traceback.print_exc()
        finally:
            try:
                unwrapped_model.audio_vae = None
                if prev_training:
                    unwrapped_model.train()
                else:
                    unwrapped_model.eval()
            except Exception as e:
                log(f"[Warning] Failed to restore model state: {e}")


def compute_mel_spectrogram(audio_np, sample_rate, n_mels=128):
    """Compute Mel Spectrogram (dB) using librosa"""
    import numpy as np
    import librosa

    audio_np = audio_np.flatten().astype(np.float32)
    mel = librosa.feature.melspectrogram(y=audio_np, sr=sample_rate, n_mels=n_mels, fmax=sample_rate // 2)
    return librosa.power_to_db(mel, ref=np.max)


def create_mel_figure(gen_audio_np, gen_mel, sample_rate, step=None, ref_audio_np=None, ref_mel=None):
    """
    Create mel spectrogram figure: show comparison if reference audio exists, otherwise show generated only
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import librosa.display

    fmax = sample_rate // 2
    step_str = f" @ Step {step}" if step is not None else ""

    if ref_audio_np is not None and ref_mel is not None:
        # Comparison mode: reference vs generated
        fig, (ax_ref, ax_gen) = plt.subplots(2, 1, figsize=(12, 8))

        img_ref = librosa.display.specshow(
            ref_mel, sr=sample_rate, x_axis="time", y_axis="mel", fmax=fmax, cmap="viridis", ax=ax_ref
        )
        ax_ref.set_title(
            f"Reference (GT) - {len(ref_audio_np)/sample_rate:.2f}s{step_str}",
            fontsize=10,
            fontweight="bold",
            color="#28A745",
        )
        plt.colorbar(img_ref, ax=ax_ref, format="%+2.0f dB", pad=0.02)

        img_gen = librosa.display.specshow(
            gen_mel, sr=sample_rate, x_axis="time", y_axis="mel", fmax=fmax, cmap="viridis", ax=ax_gen
        )
        ax_gen.set_title(
            f"Generated - {len(gen_audio_np)/sample_rate:.2f}s", fontsize=10, fontweight="bold", color="#DC3545"
        )
        plt.colorbar(img_gen, ax=ax_gen, format="%+2.0f dB", pad=0.02)
    else:
        # Single figure mode: show generated only
        fig, ax = plt.subplots(figsize=(12, 4))
        img = librosa.display.specshow(
            gen_mel, sr=sample_rate, x_axis="time", y_axis="mel", fmax=fmax, cmap="viridis", ax=ax
        )
        ax.set_title(f"Generated - {len(gen_audio_np)/sample_rate:.2f}s{step_str}", fontsize=11, fontweight="bold")
        plt.colorbar(img, ax=ax, format="%+2.0f dB", pad=0.02)

    plt.tight_layout()
    return fig


def normalize_audio(audio_np):
    """Normalize audio to [-0.9, 0.9]"""
    import numpy as np

    max_val = np.abs(audio_np).max()
    return audio_np / max_val * 0.9 if max_val > 0 else audio_np


def load_checkpoint(model, optimizer, scheduler, save_dir: Path, rank: int = 0):
    """
    Load the latest valid checkpoint if it exists.
    Called by all ranks so that distributed state stays aligned.
    Returns the step number to resume from, or 0 if no checkpoint found.
    """
    unwrapped = model.module if hasattr(model, "module") else model
    lora_cfg = unwrapped.lora_config

    for checkpoint_dir in _checkpoint_candidates(save_dir, rank=rank):
        resume_step = _try_load_checkpoint_dir(checkpoint_dir, unwrapped, lora_cfg, optimizer, scheduler, rank=rank)
        if resume_step is not None:
            return resume_step

    return 0


def _checkpoint_candidates(save_dir: Path, rank: int = 0):
    seen = set()

    def _add(candidate: Path):
        resolved = candidate.resolve()
        if resolved in seen:
            return
        seen.add(resolved)
        yield candidate

    latest_meta = save_dir / "latest.json"
    if latest_meta.exists():
        try:
            with open(latest_meta, "r", encoding="utf-8") as f:
                meta = json.load(f)
            checkpoint_dir = Path(meta.get("checkpoint_dir", ""))
            if not checkpoint_dir.is_absolute():
                checkpoint_dir = save_dir / checkpoint_dir
            if checkpoint_dir.is_dir():
                yield from _add(checkpoint_dir)
            elif rank == 0:
                print(f"Warning: latest checkpoint pointer is invalid: {checkpoint_dir}", file=sys.stderr)
        except Exception as e:
            if rank == 0:
                print(f"Warning: failed to read latest checkpoint pointer {latest_meta}: {e}", file=sys.stderr)

    if save_dir.exists():
        step_dirs = []
        for folder in save_dir.iterdir():
            if folder.is_dir() and folder.name.startswith("step_"):
                step = _checkpoint_step(folder)
                if step is not None:
                    step_dirs.append((step, folder))
        for _, folder in sorted(step_dirs, reverse=True):
            yield from _add(folder)

    legacy_latest = save_dir / "latest"
    if legacy_latest.is_dir():
        yield from _add(legacy_latest)
    elif legacy_latest.exists() and rank == 0:
        print(f"Warning: ignoring non-directory legacy latest checkpoint path: {legacy_latest}", file=sys.stderr)


def _checkpoint_step(checkpoint_dir: Path):
    state_path = checkpoint_dir / "training_state.json"
    if state_path.exists():
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            return int(state.get("step", 0))
        except Exception:
            pass

    if checkpoint_dir.name.startswith("step_"):
        try:
            return int(checkpoint_dir.name.split("_")[1])
        except (IndexError, ValueError):
            return None
    return None


def _try_load_checkpoint_dir(checkpoint_dir: Path, unwrapped, lora_cfg, optimizer, scheduler, rank: int = 0):
    if lora_cfg is not None:
        weights_path = checkpoint_dir / "lora_weights.safetensors"
        if not weights_path.exists():
            weights_path = checkpoint_dir / "lora_weights.ckpt"
        weights_kind = "LoRA weights"
    else:
        weights_path = checkpoint_dir / "model.safetensors"
        if not weights_path.exists():
            weights_path = checkpoint_dir / "pytorch_model.bin"
        weights_kind = "model weights"

    if not weights_path.exists():
        if rank == 0:
            print(f"Warning: skipping checkpoint without {weights_kind}: {checkpoint_dir}", file=sys.stderr)
        return None

    if weights_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state_dict = load_file(str(weights_path))
    else:
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=True)
        state_dict = ckpt.get("state_dict", ckpt)

    unwrapped.load_state_dict(state_dict, strict=False)
    if rank == 0:
        print(f"Loaded {weights_kind} from {weights_path}", file=sys.stderr)

    optimizer_path = checkpoint_dir / "optimizer.pth"
    if optimizer_path.exists():
        optimizer.load_state_dict(torch.load(optimizer_path, map_location="cpu", weights_only=True))
        if rank == 0:
            print(f"Loaded optimizer state from {optimizer_path}", file=sys.stderr)
    elif rank == 0:
        print(f"Warning: optimizer state not found in checkpoint: {checkpoint_dir}", file=sys.stderr)

    scheduler_path = checkpoint_dir / "scheduler.pth"
    if scheduler_path.exists():
        scheduler.load_state_dict(torch.load(scheduler_path, map_location="cpu", weights_only=True))
        if rank == 0:
            print(f"Loaded scheduler state from {scheduler_path}", file=sys.stderr)
    elif rank == 0:
        print(f"Warning: scheduler state not found in checkpoint: {checkpoint_dir}", file=sys.stderr)

    resume_step = _checkpoint_step(checkpoint_dir)
    if resume_step is None:
        if rank == 0:
            print(f"Warning: checkpoint has no valid step metadata: {checkpoint_dir}", file=sys.stderr)
        return None

    if rank == 0:
        print(f"Resuming from step {resume_step}", file=sys.stderr)
    return resume_step


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    save_dir: Path,
    step: int,
    pretrained_path: str = None,
    hf_model_id: str = "",
    distribute: bool = False,
):
    """
    Save checkpoint with different strategies for full finetune vs LoRA:
    - Full finetune: save non-vae weights to model.safetensors (or pytorch_model.bin if safetensors unavailable)
    - LoRA: save only lora weights to lora_weights.safetensors (or lora_weights.ckpt if safetensors unavailable)
    """
    import shutil

    save_dir.mkdir(parents=True, exist_ok=True)
    tag = f"step_{step:07d}"
    folder = save_dir / tag
    folder.mkdir(parents=True, exist_ok=True)

    unwrapped = model.module if hasattr(model, "module") else model
    full_state = unwrapped.state_dict()
    lora_cfg = unwrapped.lora_config

    if lora_cfg is not None:
        # LoRA finetune: save only lora_A/lora_B weights
        state_dict = {k: v for k, v in full_state.items() if "lora_" in k}
        if SAFETENSORS_AVAILABLE:
            save_file(state_dict, folder / "lora_weights.safetensors")
        else:
            torch.save({"state_dict": state_dict}, folder / "lora_weights.ckpt")

        # Save LoRA config and base model path to a separate JSON file
        # If distribute=True, save hf_model_id; otherwise save local pretrained_path
        base_model_to_save = hf_model_id if distribute else (str(pretrained_path) if pretrained_path else None)
        lora_info = {
            "base_model": base_model_to_save,
            "lora_config": lora_cfg.model_dump() if hasattr(lora_cfg, "model_dump") else vars(lora_cfg),
        }
        with open(folder / "lora_config.json", "w", encoding="utf-8") as f:
            json.dump(lora_info, f, indent=2, ensure_ascii=False)
    else:
        # Full finetune: save non-vae weights to model.safetensors
        state_dict = {k: v for k, v in full_state.items() if not k.startswith("audio_vae.")}
        if SAFETENSORS_AVAILABLE:
            save_file(state_dict, folder / "model.safetensors")
        else:
            torch.save({"state_dict": state_dict}, folder / "pytorch_model.bin")

        # Copy config files from pretrained path
        if pretrained_path:
            pretrained_dir = Path(pretrained_path)
            files_to_copy = [
                "config.json",
                "audiovae.pth",
                "audiovae.safetensors",
                "tokenizer.json",
                "special_tokens_map.json",
                "tokenizer_config.json",
            ]
            for fname in files_to_copy:
                src = pretrained_dir / fname
                if src.exists():
                    shutil.copy2(src, folder / fname)

    torch.save(optimizer.state_dict(), folder / "optimizer.pth")
    torch.save(scheduler.state_dict(), folder / "scheduler.pth")
    with open(folder / "training_state.json", "w", encoding="utf-8") as f:
        json.dump({"step": int(step)}, f)

    # Atomically update a lightweight `latest.json` pointer instead of copying
    # the whole checkpoint directory on every save.
    latest_meta = save_dir / "latest.json"
    tmp_meta = save_dir / "latest.json.tmp"
    metadata = {
        "step": int(step),
        "checkpoint_dir": tag,
        "checkpoint_format": "lora" if lora_cfg is not None else "full",
    }
    try:
        with open(tmp_meta, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        os.replace(tmp_meta, latest_meta)
    except Exception as e:
        print(f"Warning: failed to update latest checkpoint pointer at {latest_meta}: {e}", file=sys.stderr)


if __name__ == "__main__":
    from voxcpm.training.config import load_yaml_config

    args = argbind.parse_args()
    config_file = args.get("config_path")
    # If YAML config provided, use YAML args to call train
    if config_file:
        yaml_args = load_yaml_config(config_file)
        train(**yaml_args)
    else:
        # Otherwise use command line args (parsed by argbind)
        with argbind.scope(args):
            train()
