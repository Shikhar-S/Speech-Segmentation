import torch
import torch.nn as nn


class GeolocationAngularLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # https://par.nsf.gov/servlets/purl/10544360
        # not used currently to make training stable
        self.earth_radius_km = 6378.1

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        prediction: (N,3) float tensor containing predicted x y z coordinates
        target: (N,2) float tensor containing latitude and longitude
        returns total angular loss
        """
        # unpack
        x_, y_, z_ = prediction.unbind(-1)
        # [-π, π]
        pred_long = torch.atan2(y_, x_)
        # [-π/2, π/2]
        pred_lat = torch.atan2(z_, torch.clamp(torch.sqrt(x_ * x_ + y_ * y_), 1e-8))
        true_lat, true_long = target.unbind(-1)
        # compute loss
        cos_val = torch.sin(true_lat) * torch.sin(pred_lat) + torch.cos(
            true_lat
        ) * torch.cos(pred_lat) * torch.cos(pred_long - true_long)
        cos_val = torch.clamp(cos_val, -1.0 + 1e-7, 1.0 - 1e-7)
        d_angular = torch.acos(cos_val)
        total_loss = torch.mean(d_angular)
        return total_loss


# TODO(shikhar): either remove or make api consistent
# class GeolocationRegressionLoss(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         # https://par.nsf.gov/servlets/purl/10544360
#         # not used currently to make training stable
#         self.earth_radius_km = 6378.1

#     def forward(
#         self,
#         pred_v: torch.Tensor,
#         true_lat: torch.Tensor,
#         true_long: torch.Tensor,
#     ) -> torch.Tensor:
#         x = torch.cos(true_lat) * torch.cos(true_long)
#         y = torch.cos(true_lat) * torch.sin(true_long)
#         z = torch.sin(true_lat)
#         true_v = torch.stack([x, y, z], dim=-1)
#         d_regression = (pred_v - true_v).pow(2).sum(dim=-1)
#         total_loss = torch.mean(d_regression)
#         return total_loss


# class GeolocationRadianRegressionLoss(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()

#     def forward(
#         self,
#         pred_lat: torch.Tensor,
#         pred_long: torch.Tensor,
#         true_lat: torch.Tensor,
#         true_long: torch.Tensor,
#     ) -> torch.Tensor:
#         d_lat = (pred_lat - true_lat).pow(2)
#         d_long = (pred_long - true_long).pow(2)
#         total_loss = torch.mean(d_lat + d_long)
#         return total_loss


class GeolocationVMFLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(
        self, mu: torch.Tensor, kappa: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        mu: (N, 3) normalized predicted mean
        kappa: (N, 1) predicted concentration
        target: (N, 3) unit vector of true lat/long
        """
        # Batch dot product mu^T * target
        dot_product = torch.sum(mu * target, dim=-1, keepdim=True)

        # Log-normalization constant for vMF on S^2:
        # log(C_3(kappa)) = log(kappa) - log(4 * pi * sinh(kappa))
        # Stable log(sinh(k)) = k + log(1 - exp(-2k)) - log(2)
        log_sinh = (
            kappa
            + torch.log(1 - torch.exp(-2 * kappa) + self.eps)
            - torch.log(torch.tensor(2.0))
        )
        log_norm_constant = (
            torch.log(kappa + self.eps)
            - torch.log(torch.tensor(4 * torch.pi))
            - log_sinh
        )

        loss = -(log_norm_constant + kappa * dot_product)
        return loss.mean()
