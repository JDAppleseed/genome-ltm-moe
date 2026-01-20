from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    from torch.utils.data import DataLoader, IterableDataset
except ModuleNotFoundError:  # pragma: no cover - handled in main()
    DataLoader = object
    IterableDataset = object


@dataclass
class Batch:
    ids: "torch.Tensor"
    labels: "torch.Tensor"


class SyntheticTokenDataset(IterableDataset):
    def __init__(self, vocab_size: int, seq_len: int, batch_tokens: int, seed: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.batch_tokens = batch_tokens
        self.seed = seed

    def __iter__(self):
        import torch

        generator = torch.Generator().manual_seed(self.seed)
        while True:
            batch = self.batch_tokens // self.seq_len
            ids = torch.randint(0, self.vocab_size, (batch, self.seq_len), generator=generator)
            yield ids


def mask_tokens(ids: torch.Tensor, mask_prob: float) -> Batch:
    import torch

    from genomeltm.models.dna_mlm import DNA_VOCAB

    labels = ids.clone()
    mask = torch.rand(ids.shape, device=ids.device) < mask_prob
    ids = ids.clone()
    ids[mask] = DNA_VOCAB["[MASK]"]
    labels[~mask] = -100
    return Batch(ids=ids, labels=labels)


def build_dataloader(cfg: Dict[str, Any], seed: int, epoch: int) -> DataLoader:
    from genomaic.data.manifest import load_manifest
    from genomaic.data.sharding import deterministic_order, epoch_seed, shard_by_rank
    from genomaic.data.stream_fastq import FastqStreamingDataset, encode_sequences
    from genomaic.train.dist import get_rank, get_world_size
    from genomeltm.models.dna_mlm import DNA_VOCAB

    data_cfg = cfg.get("data", {})
    seq_len = int(data_cfg.get("seq_len", 1024))
    micro_batch = int(cfg.get("training", {}).get("micro_batch_size", 2))

    if data_cfg.get("synthetic", True):
        dataset = SyntheticTokenDataset(len(DNA_VOCAB), seq_len, micro_batch * seq_len, epoch_seed(seed, epoch, get_rank()))
        return DataLoader(dataset, batch_size=None)

    manifest_path = data_cfg.get("manifest_path")
    if not manifest_path:
        raise ValueError("manifest_path must be provided when synthetic data is disabled")

    manifest = load_manifest(manifest_path)
    rank = get_rank()
    world_size = get_world_size()
    shard_list = shard_by_rank(manifest, rank, world_size)
    ordered = deterministic_order(shard_list, epoch=epoch, seed=seed)
    dataset = FastqStreamingDataset(ordered, max_reads=data_cfg.get("max_reads"))

    def collate(seqs: List[str]) -> Batch:
        ids = encode_sequences(seqs, DNA_VOCAB, seq_len)
        return mask_tokens(ids, mask_prob=float(data_cfg.get("mask_prob", 0.15)))

    return DataLoader(dataset, batch_size=micro_batch, collate_fn=collate)


def build_model(cfg: Dict[str, Any]) -> torch.nn.Module:
    from genomaic.models.moe_sequence import MoESequenceModel
    from genomeltm.models.dna_mlm import DNAMaskedLM, DNA_VOCAB

    model_cfg = cfg.get("model", {})
    model_type = model_cfg.get("type", "dna_mlm")
    if model_type == "dna_mlm":
        return DNAMaskedLM(
            d_model=int(model_cfg.get("d_model", 512)),
            n_layers=int(model_cfg.get("n_layers", 8)),
            n_heads=int(model_cfg.get("n_heads", 8)),
            d_ff=int(model_cfg.get("d_ff", 2048)),
        )
    if model_type == "moe_mlm":
        return MoESequenceModel(
            vocab_size=len(DNA_VOCAB),
            d_model=int(model_cfg.get("d_model", 512)),
            n_layers=int(model_cfg.get("n_layers", 4)),
            n_experts=int(model_cfg.get("n_experts", 4)),
            d_ff=int(model_cfg.get("d_ff", 1024)),
        )
    raise ValueError(f"Unsupported model type: {model_type}")


def merge_cfg(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge_cfg(out[key], value)
        else:
            out[key] = value
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="GenomAIc distributed training")
    parser.add_argument("--config", type=str, default="configs/training_scale.yaml")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    try:
        import torch
        import torch.nn.functional as F
    except ModuleNotFoundError as exc:
        raise SystemExit("PyTorch not installed. Install torch to run training.") from exc

    from genomaic.eval.expert_utilization import log_expert_utilization, summarize_expert_utilization
    from genomaic.train.deepspeed_engine import initialize_deepspeed
    from genomaic.train.dist import get_local_rank, get_rank, get_world_size, init_distributed, is_main_process, seed_all
    from genomaic.train.ds_config import build_ds_config_from_yaml, load_training_yaml, validate_moe_zero_compat
    from genomaic.train.profiling import build_profiler, now, profile_step
    from genomaic.utils.ckpt import find_latest_checkpoint, load_checkpoint, save_checkpoint, safe_symlink_latest
    from genomaic.utils.config import load_yaml

    base_cfg = load_yaml("configs/training_defaults.yaml")
    run_cfg = load_training_yaml(args.config)
    cfg = merge_cfg(base_cfg, run_cfg)

    init_distributed(cfg.get("training", {}).get("dist_backend"))
    local_rank = get_local_rank()

    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    torch.cuda.set_device(device) if torch.cuda.is_available() else None

    seed = int(cfg.get("training", {}).get("seed", 1337))
    seed_all(seed + get_rank())

    model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("training", {}).get("learning_rate", 3e-4)))

    ds_cfg = None
    engine = None
    epoch = 0
    dataloader = build_dataloader(cfg, seed=seed, epoch=epoch)
    if cfg.get("training", {}).get("deepspeed", {}).get("enabled", False):
        ds_cfg = build_ds_config_from_yaml(cfg)
        validate_moe_zero_compat(ds_cfg, cfg)
        engine, optimizer, dataloader = initialize_deepspeed(
            model=model,
            optimizer=optimizer,
            dataloader=dataloader,
            ds_config=ds_cfg,
            grad_accum=int(cfg.get("training", {}).get("grad_accum", 1)),
        )

    total_steps = int(cfg.get("training", {}).get("steps", 100))
    ckpt_dir = Path(cfg.get("training", {}).get("checkpoint_dir", "runs/ckpt"))
    profiler = build_profiler(bool(cfg.get("training", {}).get("profile", False)))

    start_step = 0
    if args.resume:
        payload = load_checkpoint(args.resume, model=model, optimizer=optimizer, engine=engine)
        start_step = int(payload.get("step", 0))

    if not args.resume and cfg.get("training", {}).get("resume_last", False):
        latest = find_latest_checkpoint(ckpt_dir)
        if latest:
            payload = load_checkpoint(latest, model=model, optimizer=optimizer, engine=engine)
            start_step = int(payload.get("step", 0))

    data_iter = iter(dataloader)
    for step in range(start_step, total_steps):
        load_start = now()
        try:
            batch_ids = next(data_iter)
        except StopIteration:
            epoch += 1
            dataloader = build_dataloader(cfg, seed=seed, epoch=epoch)
            data_iter = iter(dataloader)
            batch_ids = next(data_iter)
        if isinstance(batch_ids, Batch):
            batch = Batch(ids=batch_ids.ids.to(device), labels=batch_ids.labels.to(device))
        else:
            batch = mask_tokens(batch_ids.to(device), mask_prob=float(cfg.get("data", {}).get("mask_prob", 0.15)))
        data_wait = now() - load_start

        step_start = now()
        if cfg.get("model", {}).get("type", "dna_mlm") == "moe_mlm":
            logits, routings = model(batch.ids)
        else:
            logits = model(batch.ids)
            routings = []
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), batch.labels.view(-1), ignore_index=-100)

        if engine is not None:
            engine.backward(loss)
            engine.step()
        else:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        step_end = now()
        tokens = batch.ids.numel()
        stats = profile_step(step_start, step_end, tokens=tokens, data_time=data_wait)

        if is_main_process() and step % int(cfg.get("training", {}).get("log_every", 10)) == 0:
            print(
                f"step={step} loss={loss.item():.4f} steps/s={stats.steps_per_sec:.2f} "
                f"tokens/s={stats.tokens_per_sec:.2f} data_wait={stats.data_time:.2f}"
            )
            if routings:
                counts = sum((routing.expert_counts for routing in routings))
                util = summarize_expert_utilization(counts)
                log_items = log_expert_utilization(util)
                print("expert_util", log_items)

        if profiler is not None:
            profiler.step()

        if step % int(cfg.get("training", {}).get("checkpoint_freq", 50)) == 0 and step > 0:
            if is_main_process():
                ckpt_path = ckpt_dir / f"step_{step}.pt"
                save_checkpoint(ckpt_path, model=model, optimizer=optimizer, step=step, epoch=0, engine=engine)
                safe_symlink_latest(ckpt_dir / "latest.pt", ckpt_path)

    if profiler is not None:
        profiler.stop()


if __name__ == "__main__":
    main()
