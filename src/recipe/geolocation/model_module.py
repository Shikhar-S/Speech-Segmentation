import pyarrow.parquet as pq  # before torch
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MinMetric, MeanMetric
from src.model.powsm.powsm_model import build_powsm


def get_kv_pooling_mask(lengths: torch.Tensor) -> torch.Tensor:
    max_len = lengths.max()
    batch_size = lengths.size(0)
    mask = torch.arange(max_len, device=lengths.device).expand(
        batch_size, max_len
    ) >= lengths.unsqueeze(1)
    return mask.unsqueeze(1)  # (B, 1, T)


class GeolocationLoss(nn.Module):
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
        d_angular = torch.acos(
            torch.sin(true_lat) * torch.sin(pred_lat)
            + torch.cos(true_lat)
            * torch.cos(pred_lat)
            * torch.cos(pred_long - true_long)
        )
        total_loss = torch.mean(d_angular)
        return total_loss


class GeolocationHead(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim, 3), nn.Tanh())

    def forward(self, x):
        v = F.normalize(self.mlp(x), dim=-1)  # (B,3) on S^2
        x_, y_, z_ = v.unbind(-1)
        # [-π, π]
        lon = torch.atan2(y_, x_)
        # [-π/2, π/2]
        lat = torch.atan2(z_, torch.clamp(torch.sqrt(x_ * x_ + y_ * y_), 1e-8))
        return v, lat, lon


class PowsmGeolocationModule(LightningModule):
    def __init__(
        self,
        s2t_train_config: str,
        s2t_model_file: str,
        bpemodel: str,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.s2t_train_config = s2t_train_config
        self.s2t_model_file = s2t_model_file

        ###### Alternate ######
        # from espnet2.bin.speech2text import Speech2Text
        # self.net = Speech2Text(
        #     s2t_train_config=self.s2t_train_config,
        #     s2t_model_file=self.s2t_model_file,
        #     bpemodel=bpemodel,
        # ).s2t_model
        #########################

        self.net, self.tokenizer = build_powsm(
            config_file=self.s2t_train_config,
            model_file=self.s2t_model_file,
            bpemodel=bpemodel,
        )

        self.encoder_dim = self.net.encoder.output_size()
        self.query_vector = nn.Parameter(torch.randn(1, 1, self.encoder_dim))
        self.attentive_pooling = nn.MultiheadAttention(
            embed_dim=self.encoder_dim, num_heads=1
        )
        self.geohead = GeolocationHead(self.encoder_dim)
        self.criterion = GeolocationLoss()

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.val_loss_best = MinMetric()

    def forward(self, x: torch.Tensor, x_lengths: torch.Tensor) -> torch.Tensor:
        h, h_len = self.net.encode(x, x_lengths)  # (B, T, D), (B,)
        b, t, d = h.size()
        attn_mask = get_kv_pooling_mask(h_len)
        h = self.attentive_pooling(
            query=self.query_vector.repeat(1, b, 1),  # (1, B, D)
            key=h.transpose(0, 1),
            value=h.transpose(0, 1),
            attn_mask=attn_mask,
        )[0].squeeze(0)
        # (B, D)
        pred = self.geohead(h)  # (B, 3), (B,), (B,)  # (xyz), lat, lon
        return pred

    def model_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        audio = batch["audio"]
        lengths = batch["lengths"]
        y_lat = batch["latitude"]
        y_long = batch["longitude"]
        coordinates, pred_lat, pred_long = self.forward(audio, lengths)
        loss = self.criterion(pred_lat, pred_long, y_lat, y_long)
        return {
            "loss": loss,
            "pred_coord": coordinates,
            "targets": torch.stack([y_lat, y_long], dim=1),
            "preds": torch.stack([pred_lat, pred_long], dim=1),
        }

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
            "train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True
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
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
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
    model = PowsmGeolocationModule(
        s2t_train_config="/work/nvme/bbjs/sbharadwaj/powsm/espnet/egs2/ipapack_plus/s2t1/exp_bigpr/s2t_train_s2t_transformer_mask_norm_raw_bpe40000/config.yaml",
        s2t_model_file="/work/nvme/bbjs/sbharadwaj/powsm/espnet/egs2/ipapack_plus/s2t1/exp_bigpr/s2t_train_s2t_transformer_mask_norm_raw_bpe40000/valid.acc.ave_5best.till40epoch.pth",
        bpemodel="/work/nvme/bbjs/sbharadwaj/powsm/espnet/egs2/ipapack_plus/s2t1/data/token_list/bpe_unigram40000/bpe.model",
        optimizer=torch.optim.Adam,
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
    )
    # print(model)
    dummy_input = torch.randn(2, 16000 * 5)  # batch of 2, 5 seconds of audio at 16kHz
    dummy_lengths = torch.tensor([16000 * 5, 16000 * 5])
    output = model.training_step(
        {
            "audio": dummy_input,
            "lengths": dummy_lengths,
            "latitude": torch.tensor([0.0, 0.0]),
            "longitude": torch.tensor([0.0, 0.0]),
        },
        0,
    )
    print(output)
    print("Model forward pass successful!")
