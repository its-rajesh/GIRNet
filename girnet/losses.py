from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def ordered_si_sdr(target: Tensor, estimate: Tensor, eps: float = 1e-8) -> Tensor:
    target = target.float()
    estimate = estimate.float()
    target = target - target.mean(dim=-1, keepdim=True)
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    scale = (estimate * target).sum(dim=-1, keepdim=True) / target.square().sum(dim=-1, keepdim=True).clamp_min(eps)
    projected = scale * target
    noise = estimate - projected
    return 10.0 * torch.log10(projected.square().sum(dim=-1).clamp_min(eps) / noise.square().sum(dim=-1).clamp_min(eps))


class MultiResolutionSTFTLoss(nn.Module):
    def __init__(self, fft_sizes: list[int] | tuple[int, ...]) -> None:
        super().__init__()
        self.fft_sizes = tuple(int(x) for x in fft_sizes)

    def forward(self, target: Tensor, estimate: Tensor) -> Tensor:
        target = target.float().flatten(0, 1)
        estimate = estimate.float().flatten(0, 1)
        total = target.new_zeros(())
        for fft_size in self.fft_sizes:
            hop = fft_size // 4
            window = torch.hann_window(fft_size, device=target.device, dtype=target.dtype)
            target_stft = torch.stft(target, fft_size, hop, fft_size, window=window, return_complex=True)
            estimate_stft = torch.stft(estimate, fft_size, hop, fft_size, window=window, return_complex=True)
            target_mag = target_stft.abs().clamp_min(1e-7)
            estimate_mag = estimate_stft.abs().clamp_min(1e-7)
            spectral_convergence = (estimate_mag - target_mag).norm() / target_mag.norm().clamp_min(1e-7)
            log_magnitude = F.l1_loss(torch.log(estimate_mag), torch.log(target_mag))
            total = total + spectral_convergence + log_magnitude
        return total / len(self.fft_sizes)


def bleed_matrix_loss(prediction: Tensor, target: Tensor, floor_db: float = -60.0) -> Tensor:
    prediction = prediction.float()
    target = target.float()
    channels = target.shape[-1]
    mask = ~torch.eye(channels, device=target.device, dtype=torch.bool).unsqueeze(0).expand_as(target)
    pred_db = 20.0 * torch.log10(prediction.clamp_min(10.0 ** (floor_db / 20.0)))
    target_db = 20.0 * torch.log10(target.clamp_min(10.0 ** (floor_db / 20.0)))
    return F.mse_loss(pred_db[mask] / 40.0, target_db[mask] / 40.0)


def _rbf_kernel(x: Tensor) -> Tensor:
    distances = torch.cdist(x, x).square()
    positive = distances[distances > 0]
    bandwidth = positive.median().detach() if positive.numel() else distances.new_tensor(1.0)
    return torch.exp(-distances / (2.0 * bandwidth.clamp_min(1e-6)))


def hsic_across_nodes(latent: Tensor, max_frames: int = 64) -> Tensor:
    """Biased RBF HSIC, treating bottleneck frames as samples per example."""
    latent = latent.float()
    batch, channels, _features, frames = latent.shape
    if frames > max_frames:
        indices = torch.linspace(0, frames - 1, max_frames, device=latent.device).long()
        latent = latent.index_select(-1, indices)
        frames = max_frames
    if frames < 2:
        return latent.new_zeros(())
    centering = torch.eye(frames, device=latent.device, dtype=latent.dtype) - torch.full(
        (frames, frames), 1.0 / frames, device=latent.device, dtype=latent.dtype
    )
    total = latent.new_zeros(())
    count = 0
    for b in range(batch):
        kernels = [_rbf_kernel(latent[b, c].transpose(0, 1)) for c in range(channels)]
        for i in range(channels):
            centered_i = centering @ kernels[i] @ centering
            for j in range(i + 1, channels):
                total = total + (centered_i * kernels[j]).sum() / ((frames - 1) ** 2)
                count += 1
    return total / max(count, 1)


@dataclass
class LossWeights:
    sisdr_weight: float = 1.0
    l1_weight: float = 1.0
    mrstft_weight: float = 0.1
    adjacency_weight: float = 5.0
    hsic_weight: float = 0.01
    identity_weight: float = 1.0
    identity_threshold_db: float = -35.0
    hsic_max_frames: int = 64
    fft_sizes: tuple[int, ...] = (512, 1024, 2048)


class GIRNetLoss(nn.Module):
    def __init__(self, weights: LossWeights, graph_mode: bool) -> None:
        super().__init__()
        self.weights = weights
        self.graph_mode = bool(graph_mode)
        self.mrstft = MultiResolutionSTFTLoss(weights.fft_sizes)

    def forward(
        self,
        output: dict[str, Tensor],
        target: Tensor,
        bleed_matrix: Tensor,
        mixture: Tensor | None = None,
        bleed_level_db: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        estimate = output["estimate"].float()
        target = target.float()
        terms = {
            "sisdr": -ordered_si_sdr(target, estimate).mean(),
            "l1": F.l1_loss(estimate, target),
            "mrstft": self.mrstft(target, estimate),
            "adjacency": estimate.new_zeros(()),
            "hsic": estimate.new_zeros(()),
            "identity": estimate.new_zeros(()),
        }
        if self.graph_mode:
            terms["adjacency"] = bleed_matrix_loss(output["adjacency"], bleed_matrix)
            if self.weights.hsic_weight > 0.0:
                terms["hsic"] = hsic_across_nodes(output["latent"], self.weights.hsic_max_frames)
        if mixture is not None and bleed_level_db is not None and self.weights.identity_weight > 0.0:
            levels = bleed_level_db.to(device=estimate.device, dtype=torch.float32).reshape(-1)
            low_bleed = levels <= self.weights.identity_threshold_db
            if torch.any(low_bleed):
                per_example = torch.mean(torch.abs(estimate - mixture.float()), dim=(1, 2))
                terms["identity"] = per_example[low_bleed].mean()
        total = (
            self.weights.sisdr_weight * terms["sisdr"]
            + self.weights.l1_weight * terms["l1"]
            + self.weights.mrstft_weight * terms["mrstft"]
            + self.weights.adjacency_weight * terms["adjacency"]
            + self.weights.hsic_weight * terms["hsic"]
            + self.weights.identity_weight * terms["identity"]
        )
        terms["total"] = total
        return total, terms
