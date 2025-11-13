import torch
from src.model.powsm.frontend import DefaultFrontend


class FrontendModel(torch.nn.Module):
    """
    A naive model that only computes features using powsm's frontend,
    without any neural network. Can act as a naive baseline.

    Expected inputs
    ---------------
    x : torch.Tensor
        Shape (B, T) float tensor of mono waveforms.
    x_lengths : torch.Tensor
        Shape (B,) lengths in samples.

    Returns
    -------
    h : torch.Tensor
        Log-Mel features of shape (B, T_frames, n_mels).
    h_lens : torch.Tensor
        Shape (B,) feature lengths in frames.
    """

    def __init__(self):
        super().__init__()
        # Use the requested config for the *default* frontend
        self.frontend = DefaultFrontend(
            fs=16000,
            n_fft=512,
            win_length=400,
            hop_length=160,
            apply_stft=True,
            frontend_conf=None,  # no WPE/MVDR; just plain STFT -> LogMel
        )

    @torch.no_grad()
    def encode(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
    ):
        """
        Compute features with the default frontend.

        Parameters
        ----------
        x : Tensor (B, T)
        x_lengths : Tensor (B,)

        Returns
        -------
        h : Tensor (B, T_frames, n_mels)
        h_lens : Tensor (B,)
        """
        if x.dim() != 2:
            raise ValueError(
                f"x must be (B, T) mono waveforms; got shape {tuple(x.shape)}"
            )
        if x.dtype != torch.float32 and x.dtype != torch.float64:
            x = x.float()

        # DefaultFrontend.forward returns (features, feature_lengths)
        h, h_lens = self.frontend(x, x_lengths)
        return h, h_lens

    def encoder_output_size(self) -> int:
        return self.frontend.n_mels
