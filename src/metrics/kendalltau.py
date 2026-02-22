import torch
from torchmetrics import Metric
from scipy.stats import kendalltau
import numpy as np


class KendallTau(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        self.preds.append(preds.detach().cpu().flatten())
        self.targets.append(targets.detach().cpu().flatten())

    def compute(self):
        preds_all = torch.cat(self.preds).cpu().numpy()
        targets_all = torch.cat(self.targets).cpu().numpy()
        if len(np.unique(preds_all)) < 2 or len(np.unique(targets_all)) < 2:
            return torch.tensor(0.0, device=self.device)
        tau, _ = kendalltau(preds_all, targets_all)
        return torch.tensor(tau, dtype=torch.float32, device=self.device)
