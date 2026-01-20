from dataclasses import dataclass
from typing import List, Tuple, Optional
import random
import torch
from torch.utils.data import Dataset

from genomeltm.models.dna_mlm import DNA_VOCAB

def encode_dna(seq: str) -> torch.Tensor:
    ids = []
    for ch in seq.upper():
        ids.append(DNA_VOCAB.get(ch, DNA_VOCAB["N"]))
    return torch.tensor(ids, dtype=torch.long)

@dataclass
class WindowExample:
    ids: torch.Tensor         # (L,)
    masked_ids: torch.Tensor  # (L,)
    mask_positions: torch.Tensor  # (M,)

class GenomeSequenceWindowDataset(Dataset):
    """
    Takes a list of sequences (strings) and samples fixed-length windows.
    This is a scaffold; replace sequences list with FASTA indexing in real use.
    """
    def __init__(self, sequences: List[str], window_len: int = 2048, mask_prob: float = 0.15, seed: int = 0):
        self.sequences = sequences
        self.window_len = window_len
        self.mask_prob = mask_prob
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return 1000000  # effectively infinite sampling

    def __getitem__(self, idx: int) -> WindowExample:
        seq = self.rng.choice(self.sequences)
        if len(seq) <= self.window_len:
            start = 0
        else:
            start = self.rng.randint(0, len(seq) - self.window_len)
        window = seq[start:start+self.window_len]
        ids = encode_dna(window)

        # mask
        mask = torch.rand_like(ids.float()) < self.mask_prob
        masked_ids = ids.clone()
        masked_ids[mask] = DNA_VOCAB["[MASK]"]
        mask_positions = mask.nonzero(as_tuple=False).squeeze(-1)
        return WindowExample(ids=ids, masked_ids=masked_ids, mask_positions=mask_positions)
