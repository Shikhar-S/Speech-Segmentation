import torch
import torch.nn.functional as F
from torchmetrics import Metric


class GeolocationDistanceError(Metric):
    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(self, earth_radius_km=6378.1, **kwargs):
        super().__init__(**kwargs)
        self.r_km = earth_radius_km
        self.add_state("sum_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        px, py, pz = preds.detach().unbind(-1)
        lat, lng = targets.detach().unbind(-1)

        p_lng = torch.atan2(py, px)
        p_lat = torch.atan2(pz, (px**2 + py**2).sqrt().clamp(min=1e-8))

        cos_d = (torch.sin(lat) * torch.sin(p_lat)) + (
            torch.cos(lat) * torch.cos(p_lat) * torch.cos(p_lng - lng)
        )

        batch_errs = torch.acos(cos_d.clamp(-1.0, 1.0))

        self.sum_error += batch_errs.sum()
        self.total += preds.size(0)

    def compute(self):
        if self.total == 0:
            return torch.tensor(0.0)
        return (self.sum_error / self.total) * self.r_km


class GeolocationMissRate(Metric):
    is_differentiable = False
    higher_is_better = False
    full_state_update = True

    def __init__(self, k=1, **kwargs):
        super().__init__(**kwargs)
        self.k = k
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        self.preds.append(preds.detach().cpu())
        self.targets.append(targets.detach().cpu())

    def compute(self):
        if not self.preds:
            return torch.tensor(0.0)

        P = torch.cat(self.preds, dim=0)
        T = torch.cat(self.targets, dim=0)

        if P.numel() == 0:
            return torch.tensor(0.0)

        G, true_idx = torch.unique(
            T.round(decimals=6), sorted=True, return_inverse=True, dim=0
        )

        glat, glng = G.unbind(-1)
        G_cart = torch.stack(
            [glat.cos() * glng.cos(), glat.cos() * glng.sin(), glat.sin()], dim=-1
        )

        if self.k >= G.size(0):
            return torch.tensor(0.0)

        sim = F.normalize(P, p=2, dim=1) @ F.normalize(G_cart, p=2, dim=1).T
        _, topk_idx = torch.topk(sim, k=min(self.k, G.size(0)), dim=1)

        hits = (topk_idx == true_idx.unsqueeze(1)).any(dim=1).float().mean()
        return 100.0 * (1.0 - hits)
