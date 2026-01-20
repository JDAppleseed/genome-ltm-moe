from dataclasses import dataclass
import torch
import torch.nn as nn

@dataclass
class VariantEffectOutput:
    # Δ score for REF->ALT in context, higher magnitude means stronger predicted effect.
    delta_score: torch.Tensor          # (B,) float
    # Posterior probability of "functional impact" (task-specific; calibrate downstream)
    impact_prob: torch.Tensor          # (B,) in [0,1]
    # Uncalibrated logits (useful for calibration / loss)
    impact_logits: torch.Tensor        # (B,)
    # Optional auxiliary: per-example confidence scalar
    confidence: torch.Tensor           # (B,) in [0,1]

class VariantEffectHead(nn.Module):
    """
    Variant effect scoring head.
    Intended inputs:
      - ref_emb: embedding of sequence window with REF allele
      - alt_emb: embedding of same window with ALT allele
    Typical usage:
      delta = f(alt_emb) - f(ref_emb) (or concatenation)
    """
    def __init__(self, d_in: int, hidden: int = 1024):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        # impact classifier uses delta representation
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        # temperature parameter for calibration (fit on calibration set)
        self.log_temp = nn.Parameter(torch.zeros(()))

        # confidence head: predicts "model confidence" based on |delta| and embeddings
        self.conf_head = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, ref_emb: torch.Tensor, alt_emb: torch.Tensor) -> VariantEffectOutput:
        """
        ref_emb: (B, d_in)
        alt_emb: (B, d_in)
        """
        # delta representation
        delta_emb = alt_emb - ref_emb  # (B, d_in)

        # delta score: signed magnitude
        delta_score = self.scorer(delta_emb).squeeze(-1)  # (B,)

        # impact classification (calibrated sigmoid)
        logits = self.classifier(delta_emb).squeeze(-1)
        temp = torch.exp(self.log_temp).clamp(0.5, 5.0)
        impact_logits = logits / temp
        impact_prob = torch.sigmoid(impact_logits)

        # confidence: mapped to [0,1]; can be trained to predict correctness / abstention
        conf_logits = self.conf_head(delta_emb).squeeze(-1)
        confidence = torch.sigmoid(conf_logits)

        return VariantEffectOutput(
            delta_score=delta_score,
            impact_prob=impact_prob,
            impact_logits=impact_logits,
            confidence=confidence,
        )
