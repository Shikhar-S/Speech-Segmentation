"""L1 Classification LightningModule.

This module implements the training/evaluation recipe for L1 (native language)
classification from speech or IPA text.

The recipe is responsible for:
- Loss computation (CrossEntropyLoss)
- Label ID <-> string mapping
- Logits post-processing (argmax)
- Metric calculation (Accuracy, F1)
- Encoder freezing options
- Supporting both audio and IPA text input modes

Run main:
    python -m src.recipe.l1_classification.model_module
"""

from typing import Any, Dict, List, Literal, Optional, Sequence

import torch
import torch.nn as nn
from lightning import LightningModule
from torchmetrics import MeanMetric, MinMetric
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
)
from lightning.pytorch.utilities import grad_norm

from src.model.heads.base_head import BaseHead, TaskType
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)
# Type alias for input modes
InputType = Literal["audio", "ipa"]


class L1ClassificationModel(LightningModule):
    """LightningModule for L1 (native language) classification.

    This module supports two setups:
    1. Audio mode: Takes an audio encoder (e.g., powsm, wav2vec2phoneme) and a head.
    2. IPA mode: Takes an IPA embedding module and a head.

    The recipe handles:
    - Loss computation using CrossEntropyLoss
    - Metric tracking (accuracy, F1)
    - Encoder freezing (for audio mode)
    - Label mapping (managed by datamodule/tokenizer)
    """

    def __init__(
        self,
        net: nn.Module,
        head: BaseHead,
        num_classes: int,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        freeze_encoder: bool = True,
        id_to_label: Optional[Sequence[str]] = None,
        input_type: InputType = "audio",
    ) -> None:
        """Initialize the L1 Classification model.

        Args:
            net: The encoder module. For audio mode, this is an audio encoder
                (e.g., powsm, wav2vec2phoneme). For IPA mode, this is an IPA embedding.
            head: Callable to the partially initialize classification head module.
            num_classes: Number of L1 classes (7 for L2Arctic).
            optimizer: Optimizer class (partial).
            scheduler: Optional learning rate scheduler class (partial).
            freeze_encoder: Whether to freeze encoder weights during training.
                For IPA mode, this is typically False since the embedding is trainable.
            id_to_label: Optional list of label strings for ID-to-label conversion.
            input_type: Input mode, either "audio" or "ipa".
        """
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net", "head"])

        self.encoder = net
        self.head = head(input_dim=self.encoder.encoder_output_size())
        self.num_classes = num_classes
        self.freeze_encoder = freeze_encoder
        self.input_type: InputType = input_type

        # Label mappings (optional, injected from datamodule/tokenizer)
        self.id_to_label: Optional[List[str]] = None
        self.label_to_id: Optional[Dict[str, int]] = None
        self._set_label_mappings(id_to_label)

        # Validate head task type
        if self.head.task_type != TaskType.CLASSIFICATION:
            raise ValueError(
                f"L1ClassificationModel requires a classification head, "
                f"got {self.head.task_type}"
            )

        # Loss function (recipe's responsibility)
        self.criterion = nn.CrossEntropyLoss()

        # Freeze encoder if requested (typically for audio mode)
        if freeze_encoder:
            self.encoder.eval()
            self.encoder.requires_grad_(False)
        else:
            self.encoder.train()
            self.encoder.requires_grad_(True)

        # Metrics
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.val_loss_best = MinMetric()

        # Classification metrics
        self.train_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)
        self.test_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.test_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")

    def _set_label_mappings(self, id_to_label: Optional[Sequence[str]]) -> None:
        """Register label/id mappings (optional)."""
        if id_to_label is None:
            self.id_to_label = None
            self.label_to_id = None
            return

        labels = list(id_to_label)
        if len(labels) != self.num_classes:
            raise ValueError(
                "Length of id_to_label must match num_classes "
                f"(got {len(labels)} vs {self.num_classes})."
            )

        self.id_to_label = labels
        self.label_to_id = {label: idx for idx, label in enumerate(labels)}

    def ids_to_labels(self, label_ids: torch.Tensor) -> List[str]:
        """Convert tensor of label ids to label strings."""
        if self.id_to_label is None:
            raise RuntimeError(
                "id_to_label mapping is not provided. "
                "Please pass id_to_label when instantiating the model."
            )
        return [self.id_to_label[idx] for idx in label_ids.tolist()]

    def forward(
        self,
        input_tensor: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass: encoder -> head -> logits.

        Args:
            input_tensor: Input tensor. For audio mode, shape is (batch, samples).
                For IPA mode, shape is (batch, seq_len) with token IDs.
            input_lengths: Length tensor of shape (batch,).

        Returns:
            Logits tensor of shape (batch, num_classes).
        """
        # Encode input (both audio encoder and IPA embedding have .encode() method)
        encoder_out, encoder_out_lengths = self.encoder.encode(
            input_tensor, input_lengths
        )

        # Pass through head to get logits
        logits = self.head(encoder_out, encoder_out_lengths)

        return logits

    def _extract_batch_inputs(
        self, batch: Dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract input tensor, lengths, and target from batch based on input_type.

        Args:
            batch: Input batch dictionary.

        Returns:
            Tuple of (input_tensor, input_lengths, target).
        """
        if self.input_type == "audio":
            # Audio mode: batch contains 'speech' and 'speech_length'
            input_tensor = batch["speech"]
            input_lengths = batch["speech_length"]
        else:
            # IPA mode: batch contains 'ipa_ids' and 'lengths'
            input_tensor = batch["ipa_ids"]
            input_lengths = batch["lengths"]

        target = batch["target"]
        return input_tensor, input_lengths, target

    def model_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Shared step for train/val/test.

        Args:
            batch: Dictionary containing:
                - For audio mode: speech, speech_length, label
                - For IPA mode: ipa_ids, lengths, label

        Returns:
            Dictionary with loss, logits, predictions, and targets.
        """
        input_tensor, input_lengths, targets = self._extract_batch_inputs(batch)

        # Forward pass
        logits = self(input_tensor, input_lengths)

        # Compute loss (recipe's responsibility)
        loss = self.criterion(logits, targets)

        # Post-processing: argmax for predictions (recipe's responsibility)
        preds = torch.argmax(logits, dim=-1)

        return {
            "loss": loss,
            "logits": logits,
            "preds": preds.detach(),
            "targets": targets.detach(),
        }

    def on_before_optimizer_step(self, optimizer) -> None:
        """Log gradient norms before optimizer step."""
        target_module = self.head if self.freeze_encoder else self
        norms = grad_norm(target_module, norm_type=2)
        self.log_dict(norms)

    def on_train_start(self) -> None:
        """Reset metrics at the start of training."""
        self.train_loss.reset()
        self.val_loss.reset()
        self.test_loss.reset()
        self.val_loss_best.reset()

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Training step.

        Args:
            batch: Input batch dictionary.
            batch_idx: Batch index.

        Returns:
            Loss tensor.
        """
        out = self.model_step(batch)

        # Update metrics
        self.train_loss(out["loss"])
        self.train_acc(out["preds"], out["targets"])

        # Log metrics
        self.log(
            "train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True
        )
        self.log(
            "train/acc", self.train_acc, on_step=True, on_epoch=True, prog_bar=True
        )

        return out["loss"]

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Validation step.

        Args:
            batch: Input batch dictionary.
            batch_idx: Batch index.
        """
        out = self.model_step(batch)

        # Update metrics
        self.val_loss(out["loss"])
        self.val_acc(out["preds"], out["targets"])
        self.val_f1(out["preds"], out["targets"])

        # Log metrics
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=False)

    def on_validation_epoch_end(self) -> None:
        """Log best validation loss at epoch end."""
        loss = self.val_loss.compute()
        self.val_loss_best(loss)
        self.log(
            "val/loss_best",
            self.val_loss_best.compute(),
            sync_dist=True,
            prog_bar=True,
        )

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Test step.

        Args:
            batch: Input batch dictionary.
            batch_idx: Batch index.
        """
        out = self.model_step(batch)

        # Update metrics
        self.test_loss(out["loss"])
        self.test_acc(out["preds"], out["targets"])
        self.test_f1(out["preds"], out["targets"])

        # Log metrics
        self.log(
            "test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/f1", self.test_f1, on_step=False, on_epoch=True, prog_bar=False)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers and schedulers.

        Returns:
            Dictionary with optimizer and optionally lr_scheduler.
        """
        if self.freeze_encoder:
            # Only optimize non-encoder parameters
            optimizable_params = [
                p for n, p in self.named_parameters() if not n.startswith("encoder.")
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
    # Sanity check for both audio and IPA modes
    from functools import partial

    from src.model.heads.transformer_head import TransformerHead
    from src.model.common.ipa_embedding import IPAEmbedding

    print("L1ClassificationModel module loaded successfully.")

    # Test IPA mode
    print("\n=== Testing IPA mode ===")
    vocab_size = 100
    embedding_dim = 128
    num_classes = 7
    batch_size = 4
    seq_len = 20

    ipa_encoder = IPAEmbedding(vocab_size=vocab_size, embedding_dim=embedding_dim)
    head = partial(
        TransformerHead(
            input_dim=embedding_dim,
            output_dim=num_classes,
            d_model=64,
            nhead=2,
            num_layers=1,
        )
    )
    model = L1ClassificationModel(
        net=ipa_encoder,
        head=head,
        optimizer=partial(torch.optim.Adam, lr=1e-4),
        num_classes=num_classes,
        freeze_encoder=False,  # IPA embedding is trainable
        input_type="ipa",
    )

    # Create dummy batch
    dummy_batch = {
        "ipa_ids": torch.randint(1, vocab_size, (batch_size, seq_len)),
        "lengths": torch.tensor([20, 15, 10, 5]),
        "label": torch.randint(0, num_classes, (batch_size,)),
    }

    # Forward pass
    out = model.model_step(dummy_batch)
    print(f"Loss: {out['loss'].item():.4f}")
    print(f"Logits shape: {out['logits'].shape}")
    print(f"Predictions: {out['preds']}")

    print("\nSanity check passed!")
