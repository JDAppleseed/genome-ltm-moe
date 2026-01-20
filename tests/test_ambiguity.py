import torch
from genomeltm.models.ambiguity import AmbiguityExpert

def test_ambiguity():
    B, d, f = 4, 64, 16
    m = AmbiguityExpert(d_model=d, feat_dim=f, abstain_threshold=0.5)
    tile = torch.randn(B, d)
    feats = torch.randn(B, f)
    out = m(tile, feats)
    assert out.ambiguity_score.shape == (B,)
    assert out.abstain_mask.shape == (B,)
