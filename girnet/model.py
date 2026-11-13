from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    mode: str = "girnet"
    channels: int = 4
    base_channels: int = 32
    depth: int = 4
    kernel_size: int = 9
    bottleneck_blocks: int = 4
    attention_heads: int = 4
    edge_hidden: int = 128

    def validate(self) -> None:
        if self.mode not in {"local", "tac", "attention", "girnet"}:
            raise ValueError(f"Unknown mode: {self.mode}")
        if self.channels < 2:
            raise ValueError("GIRNet requires at least two ordered stems")
        if self.base_channels <= 0 or self.depth < 1:
            raise ValueError("base_channels and depth must be positive")


class ResidualTemporalBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int = 1) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        groups = min(8, channels)
        while channels % groups:
            groups -= 1
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, 1),
            nn.GroupNorm(groups, channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return F.gelu(x + self.net(x))


class SharedEncoder(nn.Module):
    def __init__(self, base: int, depth: int, kernel_size: int, bottleneck_blocks: int) -> None:
        super().__init__()
        self.input = nn.Conv1d(1, base, kernel_size, padding=kernel_size // 2)
        stages = []
        downs = []
        width = base
        for _ in range(depth):
            stages.append(ResidualTemporalBlock(width, kernel_size))
            downs.append(nn.Conv1d(width, width * 2, 2 * 2, stride=2, padding=1))
            width *= 2
        self.stages = nn.ModuleList(stages)
        self.downs = nn.ModuleList(downs)
        self.bottleneck = nn.Sequential(*[
            ResidualTemporalBlock(width, kernel_size, dilation=2 ** (i % 4))
            for i in range(bottleneck_blocks)
        ])
        self.output_channels = width

    def forward(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        h = self.input(x)
        skips = []
        for stage, down in zip(self.stages, self.downs):
            h = stage(h)
            skips.append(h)
            h = down(h)
        return self.bottleneck(h), skips


class SharedDecoder(nn.Module):
    def __init__(self, bottleneck_channels: int, base: int, depth: int, kernel_size: int) -> None:
        super().__init__()
        ups, fuses = [], []
        width = bottleneck_channels
        for _ in range(depth):
            next_width = width // 2
            ups.append(nn.ConvTranspose1d(width, next_width, 4, stride=2, padding=1))
            fuses.append(nn.Sequential(
                nn.Conv1d(next_width * 2, next_width, 1),
                ResidualTemporalBlock(next_width, kernel_size),
            ))
            width = next_width
        self.ups = nn.ModuleList(ups)
        self.fuses = nn.ModuleList(fuses)
        self.output = nn.Conv1d(base, 1, kernel_size, padding=kernel_size // 2)

    def forward(self, h: Tensor, skips: list[Tensor], output_length: int) -> Tensor:
        for up, fuse, skip in zip(self.ups, self.fuses, reversed(skips)):
            h = up(h)
            if h.shape[-1] != skip.shape[-1]:
                h = F.interpolate(h, size=skip.shape[-1], mode="linear", align_corners=False)
            h = fuse(torch.cat((h, skip), dim=1))
        correction = self.output(h)
        if correction.shape[-1] != output_length:
            correction = F.interpolate(correction, size=output_length, mode="linear", align_corners=False)
        return correction


class LocalExchange(nn.Module):
    def forward(self, h: Tensor) -> tuple[Tensor, Tensor]:
        batch, channels = h.shape[:2]
        adjacency = h.new_zeros(batch, channels, channels)
        return h, adjacency


class TACExchange(nn.Module):
    def __init__(self, features: int) -> None:
        super().__init__()
        self.local = nn.Conv1d(features, features, 1)
        self.global_transform = nn.Conv1d(features, features, 1)
        self.fuse = nn.Sequential(nn.Conv1d(features * 2, features, 1), nn.GELU())

    def forward(self, h: Tensor) -> tuple[Tensor, Tensor]:
        batch, channels, features, frames = h.shape
        flat = h.reshape(batch * channels, features, frames)
        local = self.local(flat).reshape(batch, channels, features, frames)
        global_feature = self.global_transform(h.mean(dim=1)).unsqueeze(1).expand(-1, channels, -1, -1)
        fused = self.fuse(torch.cat((local, global_feature), dim=2).reshape(batch * channels, features * 2, frames))
        adjacency = h.new_zeros(batch, channels, channels)
        return h + fused.reshape(batch, channels, features, frames), adjacency


class AttentionExchange(nn.Module):
    def __init__(self, features: int, heads: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(features)
        self.attention = nn.MultiheadAttention(features, heads, batch_first=True)
        self.ffn = nn.Sequential(nn.LayerNorm(features), nn.Linear(features, features * 2), nn.GELU(), nn.Linear(features * 2, features))

    def forward(self, h: Tensor) -> tuple[Tensor, Tensor]:
        batch, channels, features, frames = h.shape
        sequence = h.permute(0, 3, 1, 2).reshape(batch * frames, channels, features)
        normalized = self.norm(sequence)
        attended, weights = self.attention(normalized, normalized, normalized, need_weights=True)
        sequence = sequence + attended
        sequence = sequence + self.ffn(sequence)
        out = sequence.reshape(batch, frames, channels, features).permute(0, 2, 3, 1)
        adjacency = weights.reshape(batch, frames, channels, channels).mean(dim=1)
        eye = torch.eye(channels, device=h.device, dtype=h.dtype).unsqueeze(0)
        adjacency = adjacency * (1.0 - eye)
        return out, adjacency


class GraphExchange(nn.Module):
    """Directed graph exchange with an independently estimable bleed matrix."""

    def __init__(self, features: int, edge_hidden: int) -> None:
        super().__init__()
        self.node_projection = nn.Sequential(nn.Linear(features * 2, features), nn.GELU())
        self.edge_network = nn.Sequential(
            nn.Linear(features * 4 + 1, edge_hidden),
            nn.GELU(),
            nn.Linear(edge_hidden, edge_hidden),
            nn.GELU(),
            nn.Linear(edge_hidden, 1),
        )
        nn.init.normal_(self.edge_network[-1].weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.edge_network[-1].bias, math.log(0.1 / 0.9))
        self.message = nn.Conv1d(features, features, 1)
        self.fuse = nn.Sequential(nn.Conv1d(features * 2, features, 1), nn.GELU(), nn.Conv1d(features, features, 1))

    def predict_adjacency(self, h: Tensor) -> Tensor:
        batch, channels, features, frames = h.shape
        stats = torch.cat((h.mean(dim=-1), h.std(dim=-1, unbiased=False)), dim=-1)
        nodes = self.node_projection(stats)
        receiver = nodes.unsqueeze(2).expand(-1, -1, channels, -1)
        sender = nodes.unsqueeze(1).expand(-1, channels, -1, -1)
        cosine = F.cosine_similarity(receiver, sender, dim=-1).unsqueeze(-1)
        pair = torch.cat((receiver, sender, torch.abs(receiver - sender), receiver * sender, cosine), dim=-1)
        adjacency = torch.sigmoid(self.edge_network(pair).squeeze(-1))
        eye = torch.eye(channels, device=h.device, dtype=h.dtype).unsqueeze(0)
        return adjacency * (1.0 - eye)

    def forward(self, h: Tensor, adjacency_override: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
        batch, channels, features, frames = h.shape
        adjacency = self.predict_adjacency(h)
        used_adjacency = adjacency
        if adjacency_override is not None:
            if adjacency_override.shape != (batch, channels, channels):
                raise ValueError(
                    f"Expected adjacency override {(batch, channels, channels)}, "
                    f"got {tuple(adjacency_override.shape)}"
                )
            eye = torch.eye(channels, device=h.device, dtype=h.dtype).unsqueeze(0)
            used_adjacency = adjacency_override.to(device=h.device, dtype=h.dtype).clamp(0.0, 1.0)
            used_adjacency = used_adjacency * (1.0 - eye)

        transformed = self.message(h.reshape(batch * channels, features, frames)).reshape(batch, channels, features, frames)
        aggregate = torch.einsum("bij,bjft->bift", used_adjacency, transformed)
        fused = self.fuse(torch.cat((h, aggregate), dim=2).reshape(batch * channels, features * 2, frames))
        return h + fused.reshape(batch, channels, features, frames), adjacency, used_adjacency


class GIRNetSeparator(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.encoder = SharedEncoder(config.base_channels, config.depth, config.kernel_size, config.bottleneck_blocks)
        features = self.encoder.output_channels
        self.decoder = SharedDecoder(features, config.base_channels, config.depth, config.kernel_size)
        if config.mode == "local":
            self.exchange = LocalExchange()
        elif config.mode == "tac":
            self.exchange = TACExchange(features)
        elif config.mode == "attention":
            self.exchange = AttentionExchange(features, config.attention_heads)
        else:
            self.exchange = GraphExchange(features, config.edge_hidden)

    def forward(self, mixture: Tensor, adjacency_override: Tensor | None = None) -> dict[str, Tensor]:
        if mixture.ndim != 3 or mixture.shape[1] != self.config.channels:
            raise ValueError(f"Expected (B,{self.config.channels},T), got {tuple(mixture.shape)}")
        batch, channels, length = mixture.shape
        flat = mixture.reshape(batch * channels, 1, length)
        encoded, skips = self.encoder(flat)
        features, frames = encoded.shape[1:]
        nodes = encoded.reshape(batch, channels, features, frames)
        if self.config.mode == "girnet":
            exchanged, adjacency, used_adjacency = self.exchange(nodes, adjacency_override)
        else:
            if adjacency_override is not None:
                raise ValueError("Adjacency override is only available in girnet mode")
            exchanged, adjacency = self.exchange(nodes)
            used_adjacency = adjacency
        decoded = self.decoder(exchanged.reshape(batch * channels, features, frames), skips, length)
        correction = decoded.reshape(batch, channels, length)
        estimate = mixture + correction
        return {
            "estimate": estimate,
            "adjacency": adjacency,
            "used_adjacency": used_adjacency,
            "latent": exchanged,
            "correction": correction,
        }

    def export_config(self) -> dict:
        return asdict(self.config)
