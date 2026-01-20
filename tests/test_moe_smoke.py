import os

import pytest

torch = pytest.importorskip("torch")

import torch.distributed as dist
import torch.multiprocessing as mp

from genomaic.models.moe_sequence import MoESequenceModel


def _worker(rank: int, world_size: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29501"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    model = MoESequenceModel(vocab_size=6, d_model=32, n_layers=1, n_experts=2, d_ff=64).cuda()
    ids = torch.randint(0, 6, (4, 32), device=rank)
    logits, routings = model(ids)
    loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), ids.view(-1))
    loss.backward()

    counts = sum((routing.expert_counts for routing in routings))
    assert counts.sum().item() > 0

    dist.destroy_process_group()


def test_moe_smoke_two_gpu():
    if torch.cuda.device_count() < 2:
        pytest.skip("Requires 2 GPUs for MoE smoke test")
    mp.spawn(_worker, args=(2,), nprocs=2, join=True)
