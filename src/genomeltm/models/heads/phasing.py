from dataclasses import dataclass
import torch
import torch.nn as nn

@dataclass
class PhasingOutput:
    # Confidence that local phasing is consistent in this region
    phase_conf: torch.Tensor       # (B,)
    # Switch-error proxy logits (lower is better)
    switch_logits: torch.Tensor    # (B,)
    switch_prob: torch.Tensor      # (B,)
    # Optional: haplotype separation score (signed)
    hap_sep: torch.Tensor          # (B,)
    confidence: torch.Tensor       # (B,)

class PhasingHead(nn.Module):
    """
    Phasing head: estimates local phasing reliability based on tile/haplotype embeddings.
    """
    def __init__(self, d_in: int, hidden: int = 1024):
        super().__init__()
        self.phase_head = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.switch_head = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.hap_sep_head = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.log_temp = nn.Parameter(torch.zeros(()))
        self.conf_head = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, emb: torch.Tensor) -> PhasingOutput:
        phase_conf = torch.sigmoid(self.phase_head(emb).squeeze(-1))

        switch_logits_raw = self.switch_head(emb).squeeze(-1)
        temp = torch.exp(self.log_temp).clamp(0.5, 5.0)
        switch_logits = switch_logits_raw / temp
        switch_prob = torch.sigmoid(switch_logits)

        hap_sep = self.hap_sep_head(emb).squeeze(-1)
        confidence = torch.sigmoid(self.conf_head(emb).squeeze(-1))

        return PhasingOutput(
            phase_conf=phase_conf,
            switch_logits=switch_logits,
            switch_prob=switch_prob,
            hap_sep=hap_sep,
            confidence=confidence,
        )
