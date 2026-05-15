"""Log global_step as a callback metric for ModelCheckpoint monitoring."""

import lightning as L
import torch


class LogGlobalStep(L.Callback):
    def on_train_batch_end(self, trainer, *_):
        trainer.callback_metrics["global_step"] = torch.tensor(
            float(trainer.global_step)
        )
