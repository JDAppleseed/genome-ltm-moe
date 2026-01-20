from dataclasses import dataclass
import torch
import torch.nn as nn

@dataclass
class AmbiguityOutput:
    ambiguity_score: torch.Tensor  # (B,)
    abstain_mask: torch.Tensor     # (B,) bool

class AmbiguityExpert(nn.Module):
    """
    Estimates intrinsic ambiguity/uncertainty given tile embedding + simple evidence features.
    This is intentionally lightweight and interpretable.
    """
    def __init__(self, d_model: int, feat_dim: int = 16, abstain_threshold: float = 0.35):
        super().__init__()
        self.abstain_threshold = abstain_threshold
        self.net = nn.Sequential(
            nn.LayerNorm(d_model + feat_dim),
            nn.Linear(d_model + feat_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),  # ambiguity score in [0,1]
        )

    def forward(self, tile_emb: torch.Tensor, evidence_feats: torch.Tensor) -> AmbiguityOutput:
        """
        tile_emb: (B, d_model)
        evidence_feats: (B, feat_dim) e.g., coverage stats, conflict flags, repeat scores
        """
        x = torch.cat([tile_emb, evidence_feats], dim=-1)
        amb = self.net(x).squeeze(-1)  # (B,)
        abstain = amb > self.abstain_threshold
        return AmbiguityOutput(ambiguity_score=amb, abstain_mask=abstain)
