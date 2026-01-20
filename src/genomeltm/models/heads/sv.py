from dataclasses import dataclass
import torch
import torch.nn as nn

@dataclass
class SVOutput:
    # Breakpoint confidence score in [0,1]
    bp_conf: torch.Tensor            # (B,)
    # Predicted SV type logits (DEL/DUP/INV/TRA/INS/etc.)
    sv_type_logits: torch.Tensor     # (B, C)
    # Predicted size class logits (e.g., <50bp, 50-500, 500-5k, 5k-50k, >50k)
    size_class_logits: torch.Tensor  # (B, K)
    # Optional: predicted breakpoint offset distribution params (toy)
    bp_offset_mu: torch.Tensor       # (B,) signed
    bp_offset_sigma: torch.Tensor    # (B,) positive
    confidence: torch.Tensor         # (B,)

class StructuralVariantHead(nn.Module):
    """
    Structural variant head. Inputs should be tile embeddings + optional conflict features.
    """
    def __init__(self, d_in: int, hidden: int = 1024, sv_types: int = 5, size_classes: int = 5):
        super().__init__()
        self.bp_head = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.type_head = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, sv_types),
        )
        self.size_head = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, size_classes),
        )
        self.offset_mu = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )
        self.offset_sigma = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
            nn.Softplus(),
        )
        self.conf_head = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, emb: torch.Tensor) -> SVOutput:
        """
        emb: (B, d_in)
        """
        bp_conf = torch.sigmoid(self.bp_head(emb).squeeze(-1))
        sv_type_logits = self.type_head(emb)
        size_class_logits = self.size_head(emb)

        bp_offset_mu = self.offset_mu(emb).squeeze(-1)
        bp_offset_sigma = self.offset_sigma(emb).squeeze(-1) + 1e-6

        confidence = torch.sigmoid(self.conf_head(emb).squeeze(-1))

        return SVOutput(
            bp_conf=bp_conf,
            sv_type_logits=sv_type_logits,
            size_class_logits=size_class_logits,
            bp_offset_mu=bp_offset_mu,
            bp_offset_sigma=bp_offset_sigma,
            confidence=confidence,
        )
