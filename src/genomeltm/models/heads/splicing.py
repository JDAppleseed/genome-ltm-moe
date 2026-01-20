from dataclasses import dataclass
import torch
import torch.nn as nn

@dataclass
class SplicingOutput:
    # ΔPSI proxy (signed); larger magnitude implies stronger predicted splice change
    delta_psi: torch.Tensor           # (B,)
    # Probability of splice impact (binary proxy); calibrated
    splice_impact_prob: torch.Tensor  # (B,)
    splice_impact_logits: torch.Tensor  # (B,)
    # Optional: predicted affected junction class (multi-class), if enabled
    junction_logits: torch.Tensor     # (B, C) or None
    confidence: torch.Tensor          # (B,)

class SplicingHead(nn.Module):
    """
    Splicing impact head for sequence windows around junctions.
    """
    def __init__(self, d_in: int, hidden: int = 1024, junction_classes: int = 0):
        super().__init__()
        self.delta_net = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.log_temp = nn.Parameter(torch.zeros(()))
        self.junction_head = None
        if junction_classes and junction_classes > 0:
            self.junction_head = nn.Sequential(
                nn.LayerNorm(d_in),
                nn.Linear(d_in, hidden),
                nn.GELU(),
                nn.Linear(hidden, junction_classes),
            )
        self.conf_head = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, delta_emb: torch.Tensor) -> SplicingOutput:
        """
        delta_emb: (B, d_in) e.g., ALT-REF embedding around junction window
        """
        delta_psi = self.delta_net(delta_emb).squeeze(-1)

        logits = self.classifier(delta_emb).squeeze(-1)
        temp = torch.exp(self.log_temp).clamp(0.5, 5.0)
        splice_impact_logits = logits / temp
        splice_impact_prob = torch.sigmoid(splice_impact_logits)

        junction_logits = self.junction_head(delta_emb) if self.junction_head is not None else None

        confidence = torch.sigmoid(self.conf_head(delta_emb).squeeze(-1))

        return SplicingOutput(
            delta_psi=delta_psi,
            splice_impact_prob=splice_impact_prob,
            splice_impact_logits=splice_impact_logits,
            junction_logits=junction_logits,
            confidence=confidence,
        )
