"""Lightning style Geolocation Model.

# TODO(shikhar): Fix metric updates
This module works with both powsm and wav2vec2phoneme encoders.
Run main:
    python -m src.recipe.geolocation.model_module
"""

import pyarrow.parquet as pq  # before torch
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MinMetric, MeanMetric
from lightning.pytorch.utilities import grad_norm
from src.model.common.utils import get_kv_pooling_mask
from src.recipe.common.geolocation_loss import GeolocationAngularLoss


# TODO(shikhar): switch with attn_mlp with task_type=GEOLOCATION
class GeolocationHead(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.Tanh(),
            nn.Linear(in_dim // 2, 3),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.mlp(x)


class GeolocationVMFHead(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.Tanh(),
        )
        self.mu_layer = nn.Linear(in_dim // 2, 3)
        self.kappa_layer = nn.Linear(in_dim // 2, 1)

    def forward(self, x):
        feat = self.backbone(x)
        mu = self.mu_layer(feat)
        mu = F.normalize(mu, p=2, dim=-1)  # Project to unit sphere

        # Use softplus to ensure kappa > 0.
        # Adding 1.0 makes kappa=1 the "starting" concentration.
        kappa = F.softplus(self.kappa_layer(feat)) + 1.0
        return mu, kappa


class GeolocationModel(LightningModule):
    def __init__(
        self,
        net: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        freeze_encoder: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.net = net
        self.encoder_dim = self.net.encoder_output_size()
        self.query_vector = nn.Parameter(torch.randn(1, 1, self.encoder_dim))
        self.attentive_pooling = nn.MultiheadAttention(
            embed_dim=self.encoder_dim, num_heads=1
        )
        self.geohead = GeolocationHead(self.encoder_dim)
        self.criterion = GeolocationAngularLoss()
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            self.net.eval()
            self.net.requires_grad_(False)
        else:
            self.net.train()
            self.net.requires_grad_(True)

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.val_loss_best = MinMetric()

    def forward(self, x: torch.Tensor, x_lengths: torch.Tensor) -> torch.Tensor:
        h, h_len = self.net.encode(x, x_lengths)  # (B, T, D), (B,)
        h = F.normalize(h, dim=-1, eps=1e-8)
        b, t, d = h.size()
        key_mask = get_kv_pooling_mask(h_len)
        q = F.normalize(self.query_vector, dim=-1, eps=1e-8).expand(
            1, b, -1
        )  # (1, B, D)
        h = self.attentive_pooling(
            query=q,  # (1, B, D)
            key=h.transpose(0, 1),
            value=h.transpose(0, 1),
            key_padding_mask=key_mask,
        )[0].squeeze(0)
        h = F.normalize(h, dim=-1, eps=1e-8)
        # (B, D)
        pred = self.geohead(h)  # (B, 3) - xyz
        return pred

    def model_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        speech = batch["speech"]
        speech_length = batch["speech_length"]
        target = batch["target"]  # (B,2) : lat, long
        coordinates = self(speech, speech_length)
        loss = self.criterion(coordinates, target)
        return {
            "loss": loss,
            "targets": target,
            "preds": coordinates.detach(),
        }

    def on_before_optimizer_step(self, optimizer):
        norms = grad_norm(self, norm_type=2)
        self.log_dict(norms)

    def on_train_start(self) -> None:
        self.train_loss.reset()
        self.test_loss.reset()
        self.val_loss.reset()
        self.val_loss_best.reset()

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        batch = self.model_step(batch)
        self.train_loss(batch["loss"])
        self.log(
            "train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True
        )
        return batch["loss"]

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        batch = self.model_step(batch)
        self.val_loss(batch["loss"])
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        loss = self.val_loss.compute()
        self.val_loss_best(loss)
        self.log(
            "val/loss_best", self.val_loss_best.compute(), sync_dist=True, prog_bar=True
        )  # important: log through compute, and sync_dist

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        batch = self.model_step(batch)
        self.test_loss(batch["loss"])
        self.log(
            "test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True
        )

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizable_params = []
        for n, p in self.named_parameters():
            if n.startswith("net") and self.freeze_encoder:
                p.requires_grad = False
                continue
            optimizable_params.append(p)
        optimizer = self.hparams.optimizer(params=optimizable_params)
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    from src.model.powsm.powsm_model import build_powsm
    from src.model.wav2vec2phoneme.wav2vec2phoneme_model import Wav2Vec2PhonemeModel

    model = GeolocationModel(
        net=Wav2Vec2PhonemeModel(
            "ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns"
        ),
        # model=build_powsm(
        #     work_dir="/work/nvme/bbjs/sbharadwaj/powsm/PhoneBench/exp/powsm_cache",
        #     hf_repo="espnet/powsm",
        # ),
        optimizer=torch.optim.Adam,
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
        freeze_encoder=True,
    )
