from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from genomaic.retrieval.index import CombinedRetriever, RetrievalResult, TileCoordinate


@dataclass(frozen=True)
class EscalationConfig:
    learned_top_k_multiplier: int
    deterministic_neighbor_add: int
    fusion_mode_on_escalate: Optional[str] = None


def apply_retrieval_escalation(
    retriever: CombinedRetriever,
    query_coord: TileCoordinate,
    escalation_level: int,
    query_embedding: Optional[np.ndarray],
    config: EscalationConfig,
) -> dict:
    learned_multiplier = max(1, config.learned_top_k_multiplier * max(1, escalation_level))
    deterministic_add = config.deterministic_neighbor_add * max(1, escalation_level)
    retrieved = retriever.retrieve_with_escalation(
        query_tile=query_coord,
        query_embedding=query_embedding,
        learned_multiplier=learned_multiplier,
        deterministic_neighbor_add=deterministic_add,
    )
    return {
        "retrieved": retrieved,
        "fusion_mode": config.fusion_mode_on_escalate,
    }
