from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT / "src"))
sys.path.append(str(REPO_ROOT))

from genomaic.data.tiles import TileSpec, iter_tiles  # noqa: E402
from genomaic.pipeline.tile_embedder import TileEmbedder, TileEmbedderConfig  # noqa: E402
from genomaic.retrieval.index import StubANNIndex  # noqa: E402

try:
    import torch
except Exception as exc:  # noqa: BLE001
    pytest.skip(f"torch unavailable: {exc}", allow_module_level=True)

from genomaic.models.local_encoder import LocalEncoder, LocalEncoderConfig  # noqa: E402
from genomaic.models.fusion import CrossTileFusion, FusionConfig  # noqa: E402
from genomaic.models.sv_evidence import SVEvidenceConfig, SVEvidenceHead  # noqa: E402


def test_end_to_end_tiny() -> None:
    contigs = {"chr1": 30}
    spec = TileSpec(
        tile_bp=10,
        stride_bp=5,
        reference_build="hg38",
        include_chrM=True,
        include_contigs=None,
        exclude_contigs=[],
    )
    tiles = list(iter_tiles(contigs, spec))
    tile_indices = [tile.tile_index for tile in tiles]

    embedder = TileEmbedder(TileEmbedderConfig(embedding_dim=4, seed=1))
    embeddings = embedder.embed_tile_indices(tile_indices).numpy()

    index = StubANNIndex(dim=4)
    index.add(embeddings, tile_indices)
    index.build()
    results = index.query(embeddings[0], top_k=1)
    assert results[0].tile_index == tile_indices[0]

    encoder = LocalEncoder(LocalEncoderConfig(d_model=4, n_layers=1, mode="ssm_stub", pool="mean"))
    encoded = encoder(torch.randn(2, 6, 4))
    fusion = CrossTileFusion(FusionConfig(d_model=4, mode="mean", dropout=0.0))
    fused = fusion(encoded.unsqueeze(1))

    head = SVEvidenceHead(SVEvidenceConfig(d_model=4, num_classes=2))
    outputs = head(fused)
    assert "logits" in outputs and "confidence" in outputs
