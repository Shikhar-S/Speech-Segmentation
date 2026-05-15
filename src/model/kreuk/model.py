"""Vendored Kreuk UnsupSeg inference-only model.

Copied near-verbatim from github.com/felixkreuk/UnsupSeg.
Only training-related code is removed.
"""

from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class NextFrameClassifier(nn.Module):
    """Kreuk UnsupSeg next-frame prediction model."""

    def __init__(self, hp):
        super().__init__()
        self.hp = hp

        Z_DIM = hp.z_dim
        LS = hp.latent_dim if hp.latent_dim != 0 else Z_DIM

        self.enc = nn.Sequential(
            nn.Conv1d(
                1, LS, kernel_size=10, stride=5, padding=0, bias=False
            ),
            nn.BatchNorm1d(LS),
            nn.LeakyReLU(),
            nn.Conv1d(
                LS, LS, kernel_size=8, stride=4, padding=0, bias=False
            ),
            nn.BatchNorm1d(LS),
            nn.LeakyReLU(),
            nn.Conv1d(
                LS, LS, kernel_size=4, stride=2, padding=0, bias=False
            ),
            nn.BatchNorm1d(LS),
            nn.LeakyReLU(),
            nn.Conv1d(
                LS, LS, kernel_size=4, stride=2, padding=0, bias=False
            ),
            nn.BatchNorm1d(LS),
            nn.LeakyReLU(),
            nn.Conv1d(
                LS, Z_DIM, kernel_size=4, stride=2, padding=0, bias=False
            ),
            LambdaLayer(lambda x: x.transpose(1, 2)),
        )

        if self.hp.z_proj != 0:
            if self.hp.z_proj_linear:
                self.enc.add_module(
                    "z_proj",
                    nn.Sequential(
                        nn.Dropout2d(self.hp.z_proj_dropout),
                        nn.Linear(Z_DIM, self.hp.z_proj),
                    ),
                )
            else:
                self.enc.add_module(
                    "z_proj",
                    nn.Sequential(
                        nn.Dropout2d(self.hp.z_proj_dropout),
                        nn.Linear(Z_DIM, Z_DIM),
                        nn.LeakyReLU(),
                        nn.Dropout2d(self.hp.z_proj_dropout),
                        nn.Linear(Z_DIM, self.hp.z_proj),
                    ),
                )

        self.pred_steps = list(
            range(
                1 + self.hp.pred_offset,
                1 + self.hp.pred_offset + self.hp.pred_steps,
            )
        )

    def score(self, f, b):
        return F.cosine_similarity(f, b, dim=-1) * self.hp.cosine_coef

    def forward(self, spect):
        z = self.enc(spect.unsqueeze(1))

        preds = defaultdict(list)
        for i, t in enumerate(self.pred_steps):
            pos_pred = self.score(z[:, :-t], z[:, t:])
            preds[t].append(pos_pred)

            for _ in range(self.hp.n_negatives):
                time_reorder = torch.arange(pos_pred.shape[1])
                batch_reorder = torch.arange(pos_pred.shape[0])
                neg_pred = self.score(
                    z[:, :-t], z[batch_reorder][:, time_reorder]
                )
                preds[t].append(neg_pred)

        return preds
