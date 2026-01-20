import pytest

torch = pytest.importorskip("torch")

from genomeltm.models.dna_mlm import DNAMaskedLM, DNA_VOCAB

def test_shapes():
    model = DNAMaskedLM(d_model=128, n_layers=2, n_heads=4, d_ff=256)
    ids = torch.full((2, 128), DNA_VOCAB["[MASK]"], dtype=torch.long)
    logits = model(ids)
    assert logits.shape == (2, 128, 6)
