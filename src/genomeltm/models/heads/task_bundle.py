from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .heads import (
    VariantEffectHead, SplicingHead, RegulatoryHead,
    StructuralVariantHead, PhasingHead, ReliabilityHead
)


@dataclass
class GenomAIcOutputs:
    variant: Optional[Any] = None
    splicing: Optional[Any] = None
    regulatory: Optional[Any] = None
    sv: Optional[Any] = None
    phasing: Optional[Any] = None
    reliability: Optional[Any] = None


class TaskBundle(nn.Module):
    """
    Drop-in module: attach to a backbone that emits token reps.

    Expected backbone outputs:
      - x: [B, L, D]  (token/tile representations)
      - mask: [B, L] bool (optional)
    Optional for variant effect:
      - ref_x: [B,L,D], alt_x: [B,L,D] if you can compute both.
    """
    def __init__(self, d_model: int, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg

        # Enable flags (defaults to True if missing)
        self.enable_variant = cfg.get("variant_effect", {}).get("enabled", True)
        self.enable_splicing = cfg.get("splicing", {}).get("enabled", True)
        self.enable_regulatory = cfg.get("regulatory", {}).get("enabled", True)
        self.enable_sv = cfg.get("sv", {}).get("enabled", True)
        self.enable_phasing = cfg.get("phasing", {}).get("enabled", True)
        self.enable_reliability = cfg.get("reliability", {}).get("enabled", True)

        if self.enable_variant:
            ve = cfg.get("variant_effect", {})
            self.variant_head = VariantEffectHead(
                d_model=d_model,
                hidden=ve.get("hidden", 1024),
                out_dim=ve.get("out_dim", 1),
                per_position=ve.get("per_position", False),
                dropout=ve.get("dropout", 0.1),
            )

        if self.enable_splicing:
            sp = cfg.get("splicing", {})
            self.splicing_head = SplicingHead(
                d_model=d_model,
                hidden=sp.get("hidden", 1024),
                out_dim=sp.get("out_dim", 1),
                dropout=sp.get("dropout", 0.1),
            )

        if self.enable_regulatory:
            rg = cfg.get("regulatory", {})
            self.regulatory_head = RegulatoryHead(
                d_model=d_model,
                n_tracks=rg.get("n_tracks", 256),
                hidden=rg.get("hidden", 2048),
                dropout=rg.get("dropout", 0.1),
            )

        if self.enable_sv:
            sv = cfg.get("sv", {})
            self.sv_head = StructuralVariantHead(
                d_model=d_model,
                n_classes=sv.get("n_classes", 6),
                hidden=sv.get("hidden", 1024),
                dropout=sv.get("dropout", 0.1),
            )

        if self.enable_phasing:
            ph = cfg.get("phasing", {})
            self.phasing_head = PhasingHead(
                d_model=d_model,
                hidden=ph.get("hidden", 512),
                dropout=ph.get("dropout", 0.1),
            )

        if self.enable_reliability:
            rl = cfg.get("reliability", {})
            self.reliability_head = ReliabilityHead(
                d_model=d_model,
                hidden=rl.get("hidden", 512),
                dropout=rl.get("dropout", 0.1),
            )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        ref_x: Optional[torch.Tensor] = None,
        alt_x: Optional[torch.Tensor] = None,
    ) -> GenomAIcOutputs:
        out = GenomAIcOutputs()

        if self.enable_variant:
            if ref_x is not None and alt_x is not None:
                out.variant = self.variant_head(ref_x, alt_x, mask=mask)
            else:
                # fallback: treat x as "alt" and ref as zeros (stub)
                zeros = torch.zeros_like(x)
                out.variant = self.variant_head(zeros, x, mask=mask)

        if self.enable_splicing:
            out.splicing = self.splicing_head(x, mask=mask)

        if self.enable_regulatory:
            out.regulatory = self.regulatory_head(x, mask=mask)

        if self.enable_sv:
            out.sv = self.sv_head(x, mask=mask)

        if self.enable_phasing:
            out.phasing = self.phasing_head(x, mask=mask)

        if self.enable_reliability:
            out.reliability = self.reliability_head(x, mask=mask)

        return out

    def compute_loss(self, outputs: GenomAIcOutputs, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Loss stubs. Replace with real labels + per-task weighting.
        Expected batch keys are placeholders; you can standardize your batch schema later.
        """
        losses: Dict[str, torch.Tensor] = {}
        total = torch.tensor(0.0, device=next(self.parameters()).device)

        # Variant effect (binary or regression)
        if outputs.variant is not None and "variant_target" in batch:
            pred = outputs.variant.delta_logits
            tgt = batch["variant_target"].to(pred.device)
            if tgt.dtype in (torch.int32, torch.int64):
                lv = F.binary_cross_entropy_with_logits(pred.float(), tgt.float())
            else:
                lv = F.mse_loss(pred.float(), tgt.float())
            losses["loss_variant"] = lv
            total = total + self.cfg.get("variant_effect", {}).get("weight", 1.0) * lv

        # Splicing
        if outputs.splicing is not None and "splicing_target" in batch:
            ls = F.mse_loss(outputs.splicing.delta_psi_logits.float(), batch["splicing_target"].float().to(outputs.splicing.delta_psi_logits.device))
            losses["loss_splicing"] = ls
            total = total + self.cfg.get("splicing", {}).get("weight", 1.0) * ls

        # Regulatory (multi-track)
        if outputs.regulatory is not None and "regulatory_target" in batch:
            lr = F.mse_loss(outputs.regulatory.track_logits.float(), batch["regulatory_target"].float().to(outputs.regulatory.track_logits.device))
            losses["loss_regulatory"] = lr
            total = total + self.cfg.get("regulatory", {}).get("weight", 1.0) * lr

        # SV
        if outputs.sv is not None and "sv_bp_target" in batch and "sv_class_target" in batch:
            lbp = F.binary_cross_entropy_with_logits(outputs.sv.breakpoint_logit.float(), batch["sv_bp_target"].float().to(outputs.sv.breakpoint_logit.device))
            lcl = F.cross_entropy(outputs.sv.class_logits.float(), batch["sv_class_target"].long().to(outputs.sv.class_logits.device))
            losses["loss_sv_bp"] = lbp
            losses["loss_sv_class"] = lcl
            w = self.cfg.get("sv", {}).get("weight", 1.0)
            total = total + w * (lbp + lcl)

        # Phasing
        if outputs.phasing is not None and "phasing_target" in batch:
            lp = F.binary_cross_entropy_with_logits(outputs.phasing.phase_consistency_logit.float(), batch["phasing_target"].float().to(outputs.phasing.phase_consistency_logit.device))
            losses["loss_phasing"] = lp
            total = total + self.cfg.get("phasing", {}).get("weight", 0.5) * lp

        # Reliability (uncertainty/conflict/abstain) — placeholder supervised or self-supervised objectives
        if outputs.reliability is not None and "abstain_target" in batch:
            la = F.binary_cross_entropy_with_logits(outputs.reliability.abstain_logit.float(), batch["abstain_target"].float().to(outputs.reliability.abstain_logit.device))
            losses["loss_abstain"] = la
            total = total + self.cfg.get("reliability", {}).get("weight", 0.5) * la

        losses["loss_total"] = total
        return losses
