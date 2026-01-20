"""Training scaffold for GenomeLTM-MoE task heads."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader, Dataset

from genomeltm.models.genome_ltm_moe import GenomeLTMMoE
from genomeltm.utils.config import load_yaml


@dataclass
class SyntheticBatch:
    x: torch.Tensor
    mask: torch.Tensor
    targets: Dict[str, torch.Tensor]


class SyntheticMultiTaskDataset(Dataset):
    def __init__(self, n_samples: int, seq_len: int, d_in: int, task_cfg: Dict[str, Dict]):
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.d_in = d_in
        self.task_cfg = task_cfg

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> SyntheticBatch:  # noqa: ARG002 - synthetic
        x = torch.randn(self.seq_len, self.d_in)
        mask = torch.ones(self.seq_len, dtype=torch.bool)
        targets: Dict[str, torch.Tensor] = {}

        variant_cfg = self.task_cfg.get("variant_effect", {})
        if variant_cfg.get("enabled", True):
            per_position = variant_cfg.get("per_position", False)
            out_dim = variant_cfg.get("out_dim", 1)
            if per_position:
                shape = (self.seq_len, out_dim) if out_dim > 1 else (self.seq_len,)
                targets["variant_target"] = torch.rand(shape)
            else:
                shape = (out_dim,) if out_dim > 1 else ()
                targets["variant_target"] = torch.rand(shape)

        if self.task_cfg.get("splicing", {}).get("enabled", True):
            out_dim = self.task_cfg.get("splicing", {}).get("out_dim", 1)
            shape = (out_dim,) if out_dim > 1 else ()
            targets["splicing_target"] = torch.rand(shape)

        if self.task_cfg.get("regulatory", {}).get("enabled", True):
            n_tracks = self.task_cfg.get("regulatory", {}).get("n_tracks", 256)
            targets["regulatory_target"] = torch.rand(n_tracks)

        if self.task_cfg.get("sv", {}).get("enabled", True):
            targets["sv_bp_target"] = torch.rand(())
            n_classes = self.task_cfg.get("sv", {}).get("n_classes", 6)
            targets["sv_class_target"] = torch.randint(0, n_classes, ())

        if self.task_cfg.get("phasing", {}).get("enabled", True):
            targets["phasing_target"] = torch.rand(())

        if self.task_cfg.get("reliability", {}).get("enabled", True):
            targets["abstain_target"] = torch.rand(())

        return SyntheticBatch(x=x, mask=mask, targets=targets)


def collate(batch):
    x = torch.stack([b.x for b in batch])
    mask = torch.stack([b.mask for b in batch])
    targets: Dict[str, torch.Tensor] = {}
    for key in batch[0].targets:
        targets[key] = torch.stack([b.targets[key] for b in batch])
    return x, mask, targets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--d_in", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--task-config", type=str, default="configs/task_heads.yaml")
    parser.add_argument("--model-config", type=str, default="configs/model_genome_ltm_moe_v0.yaml")
    parser.add_argument("--output", type=str, default="runs/trait_head")
    args = parser.parse_args()

    task_cfg = load_yaml(args.task_config)
    model_cfg = load_yaml(args.model_config)
    d_model = model_cfg.get("backbone", {}).get("ssm_blocks", {}).get("d_model", 512)
    model = GenomeLTMMoE(
        d_model=d_model,
        input_dim=args.d_in,
        task_cfg_path=args.task_config,
        model_cfg_path=args.model_config,
    ).to(args.device)

    dataset = SyntheticMultiTaskDataset(
        n_samples=max(args.steps * args.batch_size, args.batch_size),
        seq_len=args.seq_len,
        d_in=args.d_in,
        task_cfg=task_cfg,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()
    iterator = iter(loader)
    for step in range(1, args.steps + 1):
        try:
            x, mask, targets = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            x, mask, targets = next(iterator)

        x = x.to(args.device)
        mask = mask.to(args.device)
        targets = {k: v.to(args.device) for k, v in targets.items()}

        outputs = model(x, mask=mask)
        losses = model.task_bundle.compute_loss(outputs["tasks"], targets)
        total_loss = losses["loss_total"]

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

        if step % 20 == 0:
            loss_items = ", ".join(f"{k}={v.item():.4f}" for k, v in losses.items())
            print(f"step={step} {loss_items}")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / "task_head_scaffold.pt"
    torch.save({"model_state": model.state_dict(), "task_config": task_cfg}, checkpoint)
    print(f"Saved scaffold checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
