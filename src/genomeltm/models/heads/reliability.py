from dataclasses import dataclass
import torch
import torch.nn as nn

@dataclass
class ReliabilityOutput:
    # Predicts whether the model should abstain (1=abstain)
    abstain_prob: torch.Tensor      # (B,)
    abstain_logits: torch.Tensor    # (B,)
    # Cross-tech conflict probability
    conflict_prob: torch.Tensor     # (B,)
    conflict_logits: torch.Tensor   # (B,)
    # Uncertainty score (0..1 proxy)
    uncertainty: torch.Tensor       # (B,)
    # A single overall confidence score (0..1)
    confidence: torch.Tensor        # (B,)

class ReliabilityHead(nn.Module):
    """
    Reliability head that can be trained to predict:
    - abstention necessity
    - cross-tech conflict
    - uncertainty/confidence proxies
    """
    def __init__(self, d_in: int, hidden: int = 512):
        super().__init__()
        self.abstain = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.conflict = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.unc = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )
        self.conf = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )
        self.log_temp = nn.Parameter(torch.zeros(()))

    def forward(self, emb: torch.Tensor) -> ReliabilityOutput:
        abstain_logits_raw = self.abstain(emb).squeeze(-1)
        conflict_logits_raw = self.conflict(emb).squeeze(-1)
        temp = torch.exp(self.log_temp).clamp(0.5, 5.0)

        abstain_logits = abstain_logits_raw / temp
        conflict_logits = conflict_logits_raw / temp

        abstain_prob = torch.sigmoid(abstain_logits)
        conflict_prob = torch.sigmoid(conflict_logits)

        uncertainty = self.unc(emb).squeeze(-1)
        confidence = self.conf(emb).squeeze(-1)

        return ReliabilityOutput(
            abstain_prob=abstain_prob,
            abstain_logits=abstain_logits,
            conflict_prob=conflict_prob,
            conflict_logits=conflict_logits,
            uncertainty=uncertainty,
            confidence=confidence,
        )
