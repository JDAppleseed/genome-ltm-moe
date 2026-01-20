import pytest

torch = pytest.importorskip("torch")

from genomeltm.models.heads import (
    VariantEffectHead, SplicingHead, RegulatoryHead,
    StructuralVariantHead, PhasingHead, ReliabilityHead
)

def test_task_heads_shapes():
    B, d = 4, 256
    ref = torch.randn(B, d)
    alt = torch.randn(B, d)
    delta = alt - ref

    veh = VariantEffectHead(d_in=d, hidden=512)
    out = veh(ref, alt)
    assert out.delta_score.shape == (B,)
    assert out.impact_prob.shape == (B,)

    sp = SplicingHead(d_in=d, hidden=512, junction_classes=3)
    out2 = sp(delta)
    assert out2.delta_psi.shape == (B,)
    assert out2.splice_impact_prob.shape == (B,)
    assert out2.junction_logits.shape == (B, 3)

    reg = RegulatoryHead(d_in=d, hidden=512, n_tracks=8)
    out3 = reg(delta)
    assert out3.effect_score.shape == (B,)
    assert out3.track_pred.shape == (B, 8)

    sv = StructuralVariantHead(d_in=d, hidden=512, sv_types=5, size_classes=5)
    out4 = sv(torch.randn(B, d))
    assert out4.bp_conf.shape == (B,)
    assert out4.sv_type_logits.shape == (B, 5)
    assert out4.size_class_logits.shape == (B, 5)

    ph = PhasingHead(d_in=d, hidden=512)
    out5 = ph(torch.randn(B, d))
    assert out5.phase_conf.shape == (B,)
    assert out5.switch_prob.shape == (B,)

    rel = ReliabilityHead(d_in=d, hidden=256)
    out6 = rel(torch.randn(B, d))
    assert out6.abstain_prob.shape == (B,)
    assert out6.conflict_prob.shape == (B,)
