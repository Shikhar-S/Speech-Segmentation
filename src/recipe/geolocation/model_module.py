"""Lightning style Geolocation Model.

This module works with both powsm and wav2vec2phoneme encoders.
Run main:
    python -m src.recipe.geolocation.model_module
"""

import pyarrow.parquet as pq  # before torch
from typing import Any, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MinMetric, MeanMetric
from lightning.pytorch.utilities import grad_norm


def get_kv_pooling_mask(lengths):
    max_len = lengths.max()
    batch_size = lengths.size(0)
    mask = torch.arange(max_len, device=lengths.device).expand(
        batch_size, max_len
    ) >= lengths.unsqueeze(1)
    return mask  # (B, T)


class GeolocationRegressionLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # https://par.nsf.gov/servlets/purl/10544360
        # not used currently to make training stable
        self.earth_radius_km = 6378.1

    def forward(
        self,
        pred_v: torch.Tensor,
        true_lat: torch.Tensor,
        true_long: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cos(true_lat) * torch.cos(true_long)
        y = torch.cos(true_lat) * torch.sin(true_long)
        z = torch.sin(true_lat)
        true_v = torch.stack([x, y, z], dim=-1)
        d_regression = (pred_v - true_v).pow(2).sum(dim=-1)
        total_loss = torch.mean(d_regression)
        return total_loss


class GeolocationAngularLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # https://par.nsf.gov/servlets/purl/10544360
        # not used currently to make training stable
        self.earth_radius_km = 6378.1

    def forward(
        self,
        pred_lat: torch.Tensor,
        pred_long: torch.Tensor,
        true_lat: torch.Tensor,
        true_long: torch.Tensor,
    ) -> torch.Tensor:
        cos_val = torch.sin(true_lat) * torch.sin(pred_lat) + torch.cos(
            true_lat
        ) * torch.cos(pred_lat) * torch.cos(pred_long - true_long)
        cos_val = torch.clamp(cos_val, -1.0 + 1e-7, 1.0 - 1e-7)
        d_angular = torch.acos(cos_val)
        total_loss = torch.mean(d_angular)
        return total_loss


class GeolocationRadianRegressionLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        pred_lat: torch.Tensor,
        pred_long: torch.Tensor,
        true_lat: torch.Tensor,
        true_long: torch.Tensor,
    ) -> torch.Tensor:
        d_lat = (pred_lat - true_lat).pow(2)
        d_long = (pred_long - true_long).pow(2)
        total_loss = torch.mean(d_lat + d_long)
        return total_loss


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
        v = self.mlp(x)
        x_, y_, z_ = v.unbind(-1)
        # [-π, π]
        lon = torch.atan2(y_, x_)
        # [-π/2, π/2]
        lat = torch.atan2(z_, torch.clamp(torch.sqrt(x_ * x_ + y_ * y_), 1e-8))
        return v, lat, lon


class GeolocationModel(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        freeze_encoder: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.net = model
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
        else:
            self.net.train()
        # self.criterion = GeolocationRegressionLoss()
        # self.criterion = GeolocationRadianRegressionLoss()

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
        pred = self.geohead(h)  # (B, 3), (B,), (B,)  # (xyz), lat, lon
        return pred

    def model_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        speech = batch["speech"]
        speech_length = batch["speech_length"]
        y_lat = batch["latitude"]
        y_long = batch["longitude"]
        coordinates, pred_lat, pred_long = self(speech, speech_length)
        loss = self.criterion(pred_lat, pred_long, y_lat, y_long)  # angular loss
        # loss = self.criterion(coordinates, y_lat, y_long)
        return {
            "loss": loss,
            "pred_coord": coordinates,
            "targets": torch.stack([y_lat, y_long], dim=1),
            "preds": torch.stack([pred_lat, pred_long], dim=1),
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
        if self.freeze_encoder:
            optimizable_params = [
                x for n, x in self.named_parameters() if not n.startswith("net.")
            ]
        else:
            optimizable_params = list(self.parameters())
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
    from src.model.wav2vec2phoneme.wav2vec2phoneme_model import build_wav2vec2phoneme

    model = GeolocationModel(
        model=build_wav2vec2phoneme("facebook/wav2vec2-lv-60-espeak-cv-ft"),
        # model=build_powsm(
        #     work_dir="/work/nvme/bbjs/sbharadwaj/powsm/PhoneBench/exp/powsm_cache",
        #     hf_repo="espnet/powsm",
        # ),
        optimizer=torch.optim.Adam,
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
    )
    dummy_input = torch.randn(2, 16000 * 5)  # batch of 2, 5 seconds of audio at 16kHz
    dummy_lengths = torch.tensor([16000 * 5, 16000 * 5])
    output = model.training_step(
        {
            "speech": dummy_input,
            "speech_length": dummy_lengths,
            "latitude": torch.tensor([0.0, 0.0]),
            "longitude": torch.tensor([0.0, 0.0]),
        },
        0,
    )
    print(output)
    print("Model forward pass successful!")
