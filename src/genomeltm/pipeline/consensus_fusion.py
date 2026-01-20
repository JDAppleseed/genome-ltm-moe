from dataclasses import dataclass
from typing import Dict
import torch

@dataclass
class TechSupport:
    dp: torch.Tensor      # (B,)
    conf: torch.Tensor    # (B,) in [0,1]

def fuse_posteriors(per_tech: Dict[str, TechSupport], prior: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    """
    Simple, conservative fusion:
    - Convert conf to odds relative to prior, multiply odds (product-of-experts),
      then convert back to posterior.
    - Clips to avoid extreme overconfidence.
    """
    device = next(iter(per_tech.values())).conf.device
    prior_t = torch.full_like(next(iter(per_tech.values())).conf, fill_value=prior, device=device)

    # prior odds
    prior_odds = prior_t / (1.0 - prior_t + eps)

    odds = prior_odds.clone()
    for tech, sup in per_tech.items():
        c = torch.clamp(sup.conf, 0.01, 0.99)
        tech_odds = c / (1.0 - c + eps)
        odds = odds * tech_odds

    post = odds / (1.0 + odds)
    # conservative clipping
    return torch.clamp(post, 0.001, 0.999)
