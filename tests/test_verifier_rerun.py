from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "src"))

from genomaic.pipeline.verifier_rerun import EscalationConfig, apply_retrieval_escalation  # noqa: E402
from genomaic.retrieval.index import (  # noqa: E402
    CombinedRetriever,
    CoordinateWindowIndex,
    DeterministicRetrievalConfig,
    LearnedRetrievalConfig,
    StubANNIndex,
    TileCoordinate,
)


def _build_retriever() -> CombinedRetriever:
    tiles = [TileCoordinate(contig="chr1", tile_index=i) for i in range(10)]
    coordinate = CoordinateWindowIndex(tiles)
    ann = StubANNIndex(dim=2)
    ann.add(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32), [0, 9])
    ann.build()
    return CombinedRetriever(
        coordinate_index=coordinate,
        ann_index=ann,
        deterministic=DeterministicRetrievalConfig(enabled=True, neighbor_tiles=1),
        learned=LearnedRetrievalConfig(enabled=True, top_k=1),
    )


def test_escalation_expands_retrieval() -> None:
    retriever = _build_retriever()
    config = EscalationConfig(learned_top_k_multiplier=2, deterministic_neighbor_add=2)

    base = apply_retrieval_escalation(
        retriever=retriever,
        query_coord=TileCoordinate(contig="chr1", tile_index=4),
        escalation_level=1,
        query_embedding=np.array([1.0, 0.0], dtype=np.float32),
        config=config,
    )
    escalated = apply_retrieval_escalation(
        retriever=retriever,
        query_coord=TileCoordinate(contig="chr1", tile_index=4),
        escalation_level=2,
        query_embedding=np.array([1.0, 0.0], dtype=np.float32),
        config=config,
    )

    assert len(escalated["retrieved"]) >= len(base["retrieved"])


def test_fusion_mode_switch() -> None:
    retriever = _build_retriever()
    config = EscalationConfig(
        learned_top_k_multiplier=2,
        deterministic_neighbor_add=2,
        fusion_mode_on_escalate="cross_attention_fusion",
    )

    result = apply_retrieval_escalation(
        retriever=retriever,
        query_coord=TileCoordinate(contig="chr1", tile_index=4),
        escalation_level=1,
        query_embedding=np.array([1.0, 0.0], dtype=np.float32),
        config=config,
    )

    assert result["fusion_mode"] == "cross_attention_fusion"


def test_order_stable() -> None:
    retriever = _build_retriever()
    config = EscalationConfig(learned_top_k_multiplier=2, deterministic_neighbor_add=2)

    result = apply_retrieval_escalation(
        retriever=retriever,
        query_coord=TileCoordinate(contig="chr1", tile_index=4),
        escalation_level=1,
        query_embedding=np.array([1.0, 0.0], dtype=np.float32),
        config=config,
    )

    indices = [item.tile_index for item in result["retrieved"]]
    assert indices == sorted(indices)
