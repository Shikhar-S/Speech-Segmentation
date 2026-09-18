"""Weight tying for the BCE / CountCTC boundary heads.

``BCEBoundaryHead.boundary_head`` (Linear D -> 1) and ``CountCTCHead.proj``
(Linear D -> 2) both implement a binary <blank> vs <boundary> decision.
Tying them lets BCE supervision (TIMIT alignments) and CountCTC supervision
(Buckeye phone counts) train a single detector.

Done as a Lightning ``Callback`` so ``SegmentRecognizeModel.__init__`` stays
untouched. ``Callback.setup`` runs before ``configure_optimizers``, so the
optimizer is built over the tied parameter set.
"""

from typing import Any

import lightning as L
import torch
import torch.nn as nn


class BoundaryFromBinary(nn.Module):
    """Read the boundary log-odds out of a 2-class projection.

    Holds a non-registered reference to the shared ``Linear(D, 2)`` so its
    parameters are owned by exactly one module (the count_ctc head) — keeps
    ``model.parameters()`` clean while gradients still flow back through it.
    """

    def __init__(self, shared_proj: nn.Module) -> None:
        super().__init__()
        # Bypass nn.Module.__setattr__ so shared_proj is NOT a submodule.
        object.__setattr__(self, "_shared_proj", shared_proj)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self._shared_proj(x)
        return z[..., 1:2] - z[..., 0:1]


class TieBCECountCTCProj(L.Callback):
    """Tie ``bce.boundary_head`` to ``count_ctc.proj`` at fit setup time.

    Looks up ``bce`` in ``pl_module.seg_losses`` and ``count_ctc`` in
    ``pl_module.pr_losses`` (or ``seg_losses``). Replaces the BCE
    ``Linear(D, 1)`` with a stateless wrapper that reads the boundary
    log-odds from the count_ctc 2-class projection.
    """

    def setup(
        self,
        trainer: Any,
        pl_module: L.LightningModule,
        stage: str,
    ) -> None:
        if stage != "fit":
            return
        if "bce" not in pl_module.seg_losses:
            raise RuntimeError(
                "TieBCECountCTCProj requires 'bce' in seg_losses."
            )
        bce = pl_module.seg_losses["bce"]
        if "count_ctc" in pl_module.pr_losses:
            cc = pl_module.pr_losses["count_ctc"]
        elif "count_ctc" in pl_module.seg_losses:
            cc = pl_module.seg_losses["count_ctc"]
        else:
            raise RuntimeError(
                "TieBCECountCTCProj requires 'count_ctc' in pr_losses or "
                "seg_losses."
            )
        bce.boundary_head = BoundaryFromBinary(cc.proj)
