"""EVD student: a streaming depth model on a DINOv3 ViT encoder.

StreamDepth predicts log depth frame by frame. Every frame it has seen may stay in its memory
together with the depth it predicted, in three tiers: the last few frames (short), every few
frames (mid) and every many frames plus the first one (long). A new frame continues the scale
of the depth in its memory, so a whole stream comes out with one (unknown) scale.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

CONFIGS = Path(__file__).resolve().parent / "configs"
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)

# Decoder and memory width for each encoder: the DPT settings Depth Anything uses with the same ViT
# (feature width, channels of the four levels, the ViT layers they come from), and memory-attention heads.
SIZES = {
    "s": {"dim": 64, "channels": (48, 96, 192, 384), "layers": (2, 5, 8, 11), "heads": 4},
    "b": {"dim": 128, "channels": (96, 192, 384, 768), "layers": (2, 5, 8, 11), "heads": 8},
}


def _interpolate(x: torch.Tensor, size: Any) -> torch.Tensor:
    return F.interpolate(x, size=tuple(size), mode="bilinear", align_corners=True)


class ContextBlock(nn.Module):
    """Queries attend to a context (which includes themselves), then an MLP."""

    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_c = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm_m = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, q: torch.Tensor, ctx: torch.Tensor, ignore: torch.Tensor | None = None) -> torch.Tensor:
        c = self.norm_c(ctx)
        q = q + self.attn(self.norm_q(q), c, c, key_padding_mask=ignore, need_weights=False)[0]
        return q + self.mlp(self.norm_m(q))


class MemoryAttention(nn.Module):
    """A frame attends to itself and to the frames in its memory; the result is added back.

    Every memory frame is grid x grid tokens that carry its features, the log depth it was
    given, and a slot embedding that says what kind of memory it is (see StreamDepth).
    The output layer starts at zero, so the module starts as an identity.
    """

    def __init__(self, dim: int, slots: int, grid: int = 8, heads: int = 4, depth: int = 2):
        super().__init__()
        self.grid = grid
        self.pos = nn.Parameter(torch.randn(grid * grid, dim) * 0.02)
        self.slot = nn.Parameter(torch.randn(slots + 1, dim) * 0.02)  # row 0 is the empty slot, never attended
        self.depth_proj = nn.Linear(1, dim)
        self.unknown = nn.Parameter(torch.zeros(dim))  # stands in for a depth that is not known
        self.blocks = nn.ModuleList(ContextBlock(dim, heads) for _ in range(depth))
        self.out = nn.Linear(dim, dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor, tokens: torch.Tensor, depth: torch.Tensor, slot: torch.Tensor) -> torch.Tensor:
        """x: (F, C, h, w), one frame each. Frame f remembers M frames: tokens (F, M, N, C),
        depth (F, M, N) pooled log depth with NaN where unknown, slot (F, M) with 0 = empty."""
        frames, C, h, w = x.shape
        N = self.grid * self.grid
        q = F.adaptive_avg_pool2d(x, self.grid).flatten(2).transpose(1, 2) + self.pos + self.unknown
        known = ~torch.isnan(depth)[..., None]
        memory = tokens + self.pos + torch.where(known, self.depth_proj(depth.nan_to_num()[..., None]), self.unknown)
        memory = memory + self.slot[slot][:, :, None]
        ctx = torch.cat([memory.flatten(1, 2), q], dim=1)
        ignore = torch.cat([(slot == 0).repeat_interleave(N, dim=1), slot.new_zeros(frames, N, dtype=torch.bool)], dim=1)
        for block in self.blocks:
            q = block(q, ctx, ignore)
        delta = self.out(q).transpose(1, 2).reshape(frames, C, self.grid, self.grid)
        return x + _interpolate(delta, (h, w))


class ResidualConvUnit(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(F.relu(self.conv1(F.relu(x))))


class Fusion(nn.Module):
    """DPT fusion block: add the skip feature, refine, upsample."""

    def __init__(self, dim: int, skip: bool = True):
        super().__init__()
        self.skip = ResidualConvUnit(dim) if skip else None
        self.refine = ResidualConvUnit(dim)
        self.out = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None, size: Any = None) -> torch.Tensor:
        if self.skip is not None:
            x = x + self.skip(skip)
        x = self.refine(x)
        if size is None:
            size = (x.shape[-2] * 2, x.shape[-1] * 2)
        return self.out(_interpolate(x, size))


class StreamDepth(nn.Module):
    """DINOv3 encoder + DPT decoder whose two coarsest levels attend to a three-tier memory.

    Input frames must already be normalized (see `normalize`). The output is log depth.
    Its scale is free on the first frame of a stream; every later frame continues the scale of
    the depth held in memory.

    Memory tiers and their slot numbers:
        short  the last `short` frames                      1 .. short        (1 = previous frame)
        mid    every `mid_every`-th frame, last `mid`       short+1 .. short+mid   (most recent first)
        long   the first frame and every `long_every`-th    short+mid+1
    Setting mid_every or long_every to 0 switches that tier off.
    """

    def __init__(
        self,
        vit: nn.Module,
        dim: int = 64,
        channels: tuple[int, int, int, int] = (48, 96, 192, 384),
        layers: tuple[int, int, int, int] = (2, 5, 8, 11),
        short: int = 8,
        mid: int = 8,
        long: int = 8,
        mid_every: int = 8,
        long_every: int = 64,
        grid: int = 8,
        heads: int = 4,
        checkpoint_vit: bool = False,
    ):
        super().__init__()
        self.vit = vit
        self.checkpoint_vit = checkpoint_vit  # recompute the encoder in the backward pass to save memory
        self.layers = layers
        self.patch_size = int(vit.config.patch_size)
        self.num_special_tokens = 1 + int(vit.config.num_register_tokens)
        embed_dim = int(vit.config.hidden_size)
        c1, c2, c3, c4 = channels

        # Reassemble the four ViT layers at 1/4, 1/8, 1/16 and 1/32 resolution.
        self.project = nn.ModuleList(nn.Conv2d(embed_dim, c, 1) for c in channels)
        self.resize = nn.ModuleList([
            nn.ConvTranspose2d(c1, c1, 4, stride=4),
            nn.ConvTranspose2d(c2, c2, 2, stride=2),
            nn.Identity(),
            nn.Conv2d(c4, c4, 3, stride=2, padding=1),
        ])
        self.reduce = nn.ModuleList(nn.Conv2d(c, dim, 3, padding=1, bias=False) for c in channels)

        self.fuse4 = Fusion(dim, skip=False)  # deepest level: nothing below it to merge
        self.fuse3, self.fuse2, self.fuse1 = (Fusion(dim) for _ in range(3))
        self.short, self.mid, self.long = short, mid, long
        self.mid_every, self.long_every = mid_every, long_every
        self.long_slot = short + mid + 1
        self.grid = grid  # a memory frame is grid x grid tokens
        self.temporal4 = MemoryAttention(dim, self.long_slot, grid, heads)
        self.temporal3 = MemoryAttention(dim, self.long_slot, grid, heads)

        self.head1 = nn.Conv2d(dim, dim // 2, 3, padding=1)
        self.head2 = nn.Sequential(
            nn.Conv2d(dim // 2, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
        )

    def encode(self, x: torch.Tensor) -> list[torch.Tensor]:
        """(F, 3, H, W) -> four (F, C, H/16, W/16) patch feature maps."""
        h, w = x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size
        hidden = self.vit.embeddings(x)
        position = self.vit.rope_embeddings(x)
        features = []
        for i, layer in enumerate(self.vit.layer):
            if self.checkpoint_vit and self.training and torch.is_grad_enabled():
                hidden = torch.utils.checkpoint.checkpoint(
                    lambda h, p, layer=layer: layer(h, position_embeddings=p), hidden, position, use_reentrant=False
                )
            else:
                hidden = layer(hidden, position_embeddings=position)
            if i in self.layers:
                tokens = self.vit.norm(hidden)[:, self.num_special_tokens :]
                features.append(tokens.transpose(1, 2).reshape(x.shape[0], -1, h, w))
        return features

    def features(self, frames: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        """(F, 3, H, W) -> the decoder's inputs, and each frame's memory tokens (F, N, C).

        The tokens depend on the frame alone, so any frame can be put into memory by itself.
        """
        l1, l2, l3, l4 = (
            reduce(resize(project(f)))
            for f, project, resize, reduce in zip(self.encode(frames), self.project, self.resize, self.reduce)
        )
        x = self.fuse4(l4, size=l3.shape[-2:])
        tokens = F.adaptive_avg_pool2d(x, self.grid).flatten(2).transpose(1, 2)
        return [l1, l2, l3, x], tokens

    def pool_depth(self, log_depth: torch.Tensor) -> torch.Tensor:
        """(B, T, H, W) log depth, NaN = unknown -> (B, T, N) cell means, NaN for empty cells."""
        known = ~torch.isnan(log_depth)
        total = F.adaptive_avg_pool2d(log_depth.nan_to_num() * known, self.grid)
        weight = F.adaptive_avg_pool2d(known.to(total.dtype), self.grid)
        return (total / weight).flatten(2)

    def _decode(
        self, levels: list[torch.Tensor], tokens: torch.Tensor, depth: torch.Tensor, slot: torch.Tensor, size: Any
    ) -> torch.Tensor:
        """Decode F frames, each with its own memory (see MemoryAttention) -> log depth (F, H, W)."""
        l1, l2, l3, x = levels
        x = self.temporal4(x, tokens, depth, slot)
        x = self.fuse3(x, l3, size=l2.shape[-2:])
        x = self.temporal3(x, tokens, depth, slot)
        x = self.fuse2(x, l2, size=l1.shape[-2:])
        x = self.fuse1(x, l1)
        return self.head2(_interpolate(self.head1(x), size))[:, 0]

    def forward(
        self, video: torch.Tensor, state: Any = None, depth: torch.Tensor | None = None, refs: Any = None
    ) -> tuple[torch.Tensor, Any]:
        """video: (B, T, 3, H, W) -> log depth (B, T, H, W) and the stream's state.

        Streaming (no `depth`): frames are decoded one by one, each remembering the depth
        predicted before it. Pass the returned state back in to continue the same stream.

        Training (`depth` given, (B, T, H, W) log depth, NaN = unknown): all frames are decoded
        at once. Frame t has the up to `short` clip frames before it in short memory, with
        `depth` as their depth. `refs` = (tokens (B, R, N, C), log depth (B, R, H, W),
        slot (B, R)) adds R frames from elsewhere in the sequence to every frame's memory;
        get their tokens from `features`. No state is returned.
        """
        B, T, _, H, W = video.shape
        if H % self.patch_size or W % self.patch_size:
            raise ValueError(f"Input size {(H, W)} must be divisible by {self.patch_size}.")
        levels, tokens = self.features(video.flatten(0, 1))
        tokens = tokens.view(B, T, *tokens.shape[1:])
        if depth is None:
            return self._stream(levels, tokens, state, (H, W))

        back = torch.arange(self.short, 0, -1, device=video.device)  # short, ..., 1 frames ago
        rows = torch.arange(T, device=video.device)[:, None] - back  # (T, short) clip frame held in each slot
        slot = torch.where(rows >= 0, back, 0).expand(B, T, self.short)
        rows = rows.clamp(min=0)
        memory, depths = tokens[:, rows], self.pool_depth(depth)[:, rows]
        if refs is not None:
            ref_tokens, ref_depth, ref_slot = refs
            memory = torch.cat([memory, ref_tokens[:, None].expand(-1, T, -1, -1, -1)], dim=2)
            depths = torch.cat([depths, self.pool_depth(ref_depth)[:, None].expand(-1, T, -1, -1)], dim=2)
            slot = torch.cat([slot, ref_slot[:, None].expand(-1, T, -1)], dim=2)
        log_depth = self._decode(levels, memory.flatten(0, 1), depths.flatten(0, 1), slot.flatten(0, 1), (H, W))
        return log_depth.view(B, T, H, W), None

    def _stream(self, levels: list[torch.Tensor], tokens: torch.Tensor, state: Any, size: Any) -> tuple[torch.Tensor, Any]:
        B, T, N, C = tokens.shape
        # Each tier is a list of (tokens (B, N, C), pooled log depth (B, N)), oldest first.
        state = dict(state) if state else {"count": 0, "short": [], "mid": [], "long": []}
        outputs = []
        for t in range(T):
            short, mid, long = state["short"], state["mid"], state["long"]
            held = long + mid + short
            slots = [self.long_slot] * len(long)
            slots += [self.short + rank for rank in range(len(mid), 0, -1)]
            slots += list(range(len(short), 0, -1))
            if held:
                memory = torch.stack([entry[0] for entry in held], dim=1)
                depths = torch.stack([entry[1] for entry in held], dim=1)
            else:
                memory, depths = tokens.new_zeros(B, 0, N, C), tokens.new_zeros(B, 0, N)
            slot = torch.tensor(slots, dtype=torch.long, device=tokens.device).expand(B, -1)
            frame = [level.view(B, T, *level.shape[1:])[:, t] for level in levels]
            log_depth = self._decode(frame, memory, depths, slot, size)
            outputs.append(log_depth)

            entry = (tokens[:, t], self.pool_depth(log_depth[:, None])[:, 0])
            count = state["count"]
            state["short"] = (short + [entry])[-self.short :] if self.short else []  # short = 0: no memory at all
            if self.mid_every and count % self.mid_every == 0:
                state["mid"] = (mid + [entry])[-self.mid :]
            if self.long_every and count % self.long_every == 0:
                long = long + [entry]
                state["long"] = long if len(long) <= self.long else long[:1] + long[2:]  # the first frame stays
            state["count"] = count + 1
        return torch.stack(outputs, dim=1), state


def normalize(frames: torch.Tensor) -> torch.Tensor:
    """(T, h, w, 3) uint8 RGB -> (T, 3, h, w) float, normalized as DINOv3 expects."""
    frames = frames.permute(0, 3, 1, 2).float() / 255.0
    return (frames - frames.new_tensor(IMAGE_MEAN)[:, None, None]) / frames.new_tensor(IMAGE_STD)[:, None, None]


def dinov3_encoder(size: str) -> nn.Module:
    """An uninitialised DINOv3 ViT-S/16 or ViT-B/16 (`size` "s" or "b"); the checkpoint fills its weights."""
    from .dinov3 import DINOv3ViTConfig, DINOv3ViTModel

    config = json.loads((CONFIGS / f"dinov3_vit{size}16.json").read_text())
    config.pop("architectures", None)
    config.pop("transformers_version", None)
    return DINOv3ViTModel(DINOv3ViTConfig(**config))


def load_model(checkpoint: str | Path, device: str = "cuda") -> tuple[StreamDepth, dict]:
    """Build the model a checkpoint was trained as and load its weights. Returns (model, training args)."""
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    args = saved["args"]
    size = args.get("size", "s")
    memory = args.get("memory", 8)
    model = StreamDepth(
        dinov3_encoder(size),
        grid=args.get("grid", 8),
        short=memory,
        mid=memory,
        long=memory,
        **SIZES[size],
    )
    model.load_state_dict(saved["model"])
    return model.to(device).eval(), args
