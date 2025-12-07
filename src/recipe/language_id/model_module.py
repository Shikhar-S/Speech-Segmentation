"""Lightning style Language Identification Model.

This module works with both powsm and wav2vec2phoneme encoders.
Run main:
    python -m src.recipe.language_id.model_module
"""

from typing import Any, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MinMetric, MeanMetric
from torchmetrics.classification import Accuracy
from lightning.pytorch.utilities import grad_norm
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def get_kv_pooling_mask(lengths):
    max_len = lengths.max()
    batch_size = lengths.size(0)
    mask = torch.arange(max_len, device=lengths.device).expand(
        batch_size, max_len
    ) >= lengths.unsqueeze(1)
    return mask  # (B, T)


class LanguageIdHead(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(in_dim // 2, num_classes),
        )

    def forward(self, x):
        logits = self.mlp(x)  # (B, num_classes)
        return logits


class LanguageIdModel(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        num_classes: Optional[int] = None,
        freeze_encoder: bool = True,
        net: Optional[
            nn.Module
        ] = None,  # Ignored - kept for Hydra config compatibility
        **kwargs,  # Accept any extra kwargs to avoid errors
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["model", "net"])
        self.net = model
        self.encoder_dim = self.net.encoder_output_size()
        # num_classes can be set from config or from datamodule later
        self.num_classes = num_classes
        self.query_vector = nn.Parameter(torch.randn(1, 1, self.encoder_dim))
        self.attentive_pooling = nn.MultiheadAttention(
            embed_dim=self.encoder_dim, num_heads=1
        )
        # Initialize head later if num_classes is not provided yet
        if num_classes is not None:
            self.langid_head = LanguageIdHead(self.encoder_dim, num_classes)
        else:
            self.langid_head = None
        self.criterion = nn.CrossEntropyLoss()
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

        # Accuracy metrics - will be initialized when num_classes is known
        self.train_acc = None
        self.val_acc = None
        self.test_acc = None
        if num_classes is not None:
            self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
            self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
            self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)

    def setup_num_classes(self, num_classes: int):
        """Set num_classes and initialize head if not already initialized.
        This can be called from the datamodule after setup() or auto-detected from batch.
        """
        if self.num_classes is None or self.num_classes != num_classes:
            self.num_classes = num_classes
            self.langid_head = LanguageIdHead(self.encoder_dim, num_classes)
            self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
            self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
            self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)
            # Lightning will handle device placement automatically

    def forward(self, x: torch.Tensor, x_lengths: torch.Tensor) -> torch.Tensor:
        log.info(f"Forward pass with input shape: {x.shape}")
        if self.langid_head is None:
            raise RuntimeError(
                "num_classes not set. Call setup_num_classes() first or provide num_classes in config."
            )
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
        logits = self.langid_head(h)  # (B, num_classes)
        return logits

    def model_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        speech = batch["speech"]
        speech_length = batch["speech_length"]
        y_lang_id = batch["lang_id"]

        # Auto-detect num_classes from batch if not set
        if self.langid_head is None:
            max_class = y_lang_id.max().item() + 1
            self.setup_num_classes(max_class)
            # Log a warning that we're using auto-detected num_classes
            if self.trainer and self.trainer.is_global_zero:
                import warnings

                warnings.warn(
                    f"num_classes was not set in config. Auto-detected {max_class} classes from batch. "
                    f"Consider setting num_classes: {max_class} in your model config for better performance."
                )

        logits = self(speech, speech_length)
        loss = self.criterion(logits, y_lang_id)
        return {
            "loss": loss,
            "logits": logits,
            "targets": y_lang_id,
            "preds": logits.argmax(dim=-1),
        }

    def on_before_optimizer_step(self, optimizer):
        norms = grad_norm(self, norm_type=2)
        self.log_dict(norms)

    def on_train_start(self) -> None:
        self.train_loss.reset()
        self.test_loss.reset()
        self.val_loss.reset()
        self.val_loss_best.reset()
        if self.train_acc is not None:
            self.train_acc.reset()
        if self.val_acc is not None:
            self.val_acc.reset()
        if self.test_acc is not None:
            self.test_acc.reset()

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        batch = self.model_step(batch)
        self.train_loss(batch["loss"])
        if self.train_acc is not None:
            self.train_acc(batch["preds"], batch["targets"])
            self.log(
                "train/acc", self.train_acc, on_step=True, on_epoch=True, prog_bar=True
            )
        self.log(
            "train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True
        )
        return batch["loss"]

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        batch = self.model_step(batch)
        self.val_loss(batch["loss"])
        if self.val_acc is not None:
            self.val_acc(batch["preds"], batch["targets"])
            self.log(
                "val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True
            )
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
        if self.test_acc is not None:
            self.test_acc(batch["preds"], batch["targets"])
            self.log(
                "test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True
            )
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
    from src.model.wav2vec2phoneme.wav2vec2phoneme_model import Wav2Vec2PhonemeModel

    # Test with dummy data
    num_classes = 102  # FLEURS has 102 languages

    model = LanguageIdModel(
        model=Wav2Vec2PhonemeModel(
            "ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns"
        ),
        # model=build_powsm(
        #     work_dir="/work/nvme/bbjs/sbharadwaj/powsm/PhoneBench/exp/powsm_cache",
        #     hf_repo="espnet/powsm",
        # ),
        optimizer=torch.optim.Adam,
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
        num_classes=num_classes,
        freeze_encoder=True,
    )

    # Test forward pass
    dummy_input = torch.randn(2, 16000 * 5)  # batch of 2, 5 seconds of audio at 16kHz
    dummy_lengths = torch.tensor([16000 * 5, 16000 * 5])
    dummy_lang_ids = torch.tensor([0, 1])

    output = model.training_step(
        {
            "speech": dummy_input,
            "speech_length": dummy_lengths,
            "lang_id": dummy_lang_ids,
        },
        0,
    )
    print(f"Model forward pass successful! Loss: {output.item()}")
