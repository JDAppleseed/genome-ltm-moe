import torch
import torch.nn as nn

class TraitPredictor(nn.Module):
    """
    Takes region/tile embeddings (from genome model) and predicts trait probabilities.
    This is a safe direction (genome -> trait). It does not generate sequences.
    """
    def __init__(self, d_in: int, n_traits: int, hidden: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_traits),
        )
        # Temperature for calibration (can be fitted on a calibration set)
        self.log_temp = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, d_in) pooled embedding (e.g., mean over relevant tiles)
        returns probs: (B, n_traits)
        """
        logits = self.net(x)
        temp = torch.exp(self.log_temp).clamp(0.5, 5.0)
        logits = logits / temp
        return torch.sigmoid(logits)
