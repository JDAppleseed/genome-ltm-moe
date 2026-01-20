from dataclasses import dataclass
import torch
import torch.nn as nn

@dataclass
class RegulatoryOutput:
    # Continuous regulatory effect score (signed)
    effect_score: torch.Tensor           # (B,)
    # Probability that this variant/region alters regulatory activity
    effect_prob: torch.Tensor            # (B,)
    effect_logits: torch.Tensor          # (B,)
    # Optional multi-task track prediction (e.g., accessibility, TF binding), if enabled
    track_pred: torch.Tensor             # (B, T) or None
    confidence: torch.Tensor             # (B,)

class RegulatoryHead(nn.Module):
    """
    Regulatory effect head. Can be used as:
    - delta_emb input (ALT-REF embedding)
    - or direct region embedding for predicting tracks.
    """
    def __init__(self, d_in: int, hidden: int = 1024, n_tracks: int = 0):
        super().__init__()
        self.effect_net = nn.Sequential(
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
        self.track_head = None
        if n_tracks and n_tracks > 0:
            self.track_head = nn.Sequential(
                nn.LayerNorm(d_in),
                nn.Linear(d_in, hidden),
                nn.GELU(),
                nn.Linear(hidden, n_tracks),
            )
        self.conf_head = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, emb: torch.Tensor) -> RegulatoryOutput:
        """
        emb: (B, d_in)
        """
        effect_score = self.effect_net(emb).squeeze(-1)

        logits = self.classifier(emb).squeeze(-1)
        temp = torch.exp(self.log_temp).clamp(0.5, 5.0)
        effect_logits = logits / temp
        effect_prob = torch.sigmoid(effect_logits)

        track_pred = self.track_head(emb) if self.track_head is not None else None
        confidence = torch.sigmoid(self.conf_head(emb).squeeze(-1))

        return RegulatoryOutput(
            effect_score=effect_score,
            effect_prob=effect_prob,
            effect_logits=effect_logits,
            track_pred=track_pred,
            confidence=confidence,
        )
