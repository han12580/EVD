"""Keep one scale over a sequence with point tracks (CoTracker3 online, forward only).

A keyframe gets a grid of points, followed until the next keyframe. A frame takes its scale from its
keyframe. Only the scale of each frame is changed; its shape stays as the depth model predicted it.

The scale step from the keyframe to a frame (`mode`):
- "3d" (default): the tracked points are lifted to 3D with the model's depth and the camera intrinsics in
  both frames, and one similarity (scale, rotation, translation) maps the frame's points onto the
  keyframe's; its scale is the step. This holds for any scene shape, a colon included.
- "2d": the distances between the tracked points grew by some factor and their depth shrank by the same
  factor. Needs no intrinsics, but only holds for a surface facing the camera: in a colon (SimCol3D,
  C3VD) near-wall and far-lumen points change depth by different ratios, and the step is biased.

Guards: a new keyframe starts as soon as fewer than `min_frac` of the points are still tracked (at the
latest `every` frames after the last one); a frame's step is used only if at least `min_frac` of the
points are tracked and the fit agrees ("3d": median residual of the kept points at most `max_resid` of
their median depth; "2d": median absolute deviation of the pairs' log growth at most `max_spread`).
Otherwise the frame keeps the scale change the depth model predicted itself. mode="2d" with min_frac=0
and max_spread=inf is the first, unguarded tracker (a keyframe every `every` frames).

Fast camera (`fast_gap`; off when None): when the camera moves fast the points are lost within a few frames, a
keyframe follows every ~10 frames, and the chain links a hundred keyframe steps or more. Each step carries the same
bias from the model's near/far shape (a compressed range under forward motion down a tube), so the chained scale
walks off (SimCol3D: by 1-4 log units; with GT depth it stays within 0.4). Fast = the median of the last 8 completed
keyframe gaps below fast_gap frames (until it rises above slow_gap again). While fast, the correction relaxes to the
model's own scale with time constant fast_tau frames: the tracks fix the short-term scale changes, the model's
memory keeps the long-term scale. Decided from past frames only.
"""

from __future__ import annotations

import math

import numpy as np
import torch



class ScaleTracker:
    def __init__(
        self,
        device: str = "cuda",
        every: int = 64,
        grid: int = 20,
        mode: str = "3d",
        min_frac: float = 0.5,
        max_spread: float = 0.05,
        max_resid: float = 0.05,
        fast_gap: float | None = None,
        slow_gap: float = 32,
        fast_tau: float = 50,
        checkpoint: str = "checkpoints/scaled_online.pth",
    ):
        from cotracker.predictor import CoTrackerOnlinePredictor

        self.model = CoTrackerOnlinePredictor(checkpoint=checkpoint).to(device)
        self.device, self.every, self.grid = device, every, grid
        self.mode, self.min_frac, self.max_spread, self.max_resid = mode, min_frac, max_spread, max_resid
        self.fast_gap, self.slow_gap, self.fast_tau = fast_gap, slow_gap, fast_tau
        self.used = float("nan")  # share of frames whose step came from the tracks, in the last sequence
        self.fast = float("nan")  # share of frames handled as a fast camera, in the last sequence

    @torch.no_grad()
    def _follow(self, rgb: np.ndarray, key: int, last: int, queries: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """The points seeded on frame `key` in frames key+1 .. last: (n, P, 2) as x, y, and (n, P) visible."""
        step = self.model.step
        n = last - key + 1

        def window(start: int) -> torch.Tensor:
            rows = (key + np.arange(start, start + 2 * step)).clip(max=last)
            frames = torch.from_numpy(np.stack([rgb[r] for r in rows]))
            return frames.permute(0, 3, 1, 2).to(self.device).float()[None]

        self.model(video_chunk=window(0), is_first_step=True, queries=queries.to(self.device)[None])
        padded = step + max(1, math.ceil((n - step) / step)) * step
        for start in range(0, padded - step, step):
            points, seen = self.model(video_chunk=window(start))
        return points[0, 1:n].cpu().numpy(), seen[0, 1:n].cpu().numpy() > 0.5

    def track(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """rgb: (T, H, W, 3) uint8. Returns the seed pixels (P, 2) as x, y, each frame's keyframe (T,), where
        its keyframe's points are in that frame (T, P, 2), and whether they are visible (T, P)."""
        T, H, W, _ = rgb.shape
        ys = torch.linspace(0.04 * H, 0.96 * H, self.grid)
        xs = torch.linspace(0.04 * W, 0.96 * W, self.grid)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        queries = torch.stack([torch.zeros(self.grid**2), gx.reshape(-1), gy.reshape(-1)], dim=1)  # frame, x, y
        P = self.grid**2
        uv = np.zeros((T, P, 2), np.float32)
        visible = np.zeros((T, P), bool)
        keys = np.zeros(T, np.int64)

        key = 0
        while key < T - 1:
            last = min(key + self.every, T - 1)
            points, seen = self._follow(rgb, key, last, queries)
            inside = seen & (points[..., 0] >= 0) & (points[..., 0] < W) & (points[..., 1] >= 0) & (points[..., 1] < H)
            lost = np.flatnonzero(inside.mean(1) < self.min_frac)
            end = last if not len(lost) else max(key + 1, key + lost[0])  # the last frame that still has enough
            uv[key + 1 : end + 1], visible[key + 1 : end + 1] = points[: end - key], seen[: end - key]
            keys[key + 1 : end + 1] = key
            key = end
        return queries[:, 1:].numpy(), keys, uv, visible

    def correct(
        self,
        log_depth: torch.Tensor,
        K: np.ndarray | None,
        seed: np.ndarray,
        keys: np.ndarray,
        uv: np.ndarray,
        visible: np.ndarray,
    ) -> torch.Tensor:
        """log_depth: (T, h, w) on the CPU, at the resolution the tracks are in; K: their 3x3 intrinsics
        (mode "3d"). Returns it with one constant added per frame so that every frame continues its
        keyframe's scale."""
        logs = log_depth.numpy()
        T, h, w = logs.shape
        rng = np.random.default_rng(0)

        def at(image: np.ndarray, points: np.ndarray) -> np.ndarray:
            x = np.clip(np.round(points[:, 0]).astype(int), 0, w - 1)
            y = np.clip(np.round(points[:, 1]).astype(int), 0, h - 1)
            return image[y, x]

        def lift(points: np.ndarray, log_z: np.ndarray) -> np.ndarray:
            rays = np.concatenate([points, np.ones((len(points), 1))], 1) @ np.linalg.inv(K).T
            return rays * np.exp(log_z)[:, None]

        shift = np.zeros(T, np.float32)
        used = 0
        for t in range(1, T):
            key = keys[t]
            ok = visible[t] & (uv[t, :, 0] >= 0) & (uv[t, :, 0] < w) & (uv[t, :, 1] >= 0) & (uv[t, :, 1] < h)
            then, now = seed[ok], uv[t][ok]
            shift[t] = shift[t - 1]  # kept when the tracks cannot be trusted: the model's own scale change
            if len(then) < max(50, self.min_frac * len(seed)):
                continue
            if self.mode == "3d":
                scale, resid = similarity(lift(then, at(logs[key], then) + shift[key]), lift(now, at(logs[t], now)))
                if resid <= self.max_resid:
                    shift[t], used = np.log(scale), used + 1
                continue
            i, j = rng.integers(0, len(then), 4000), rng.integers(0, len(then), 4000)
            before, after = np.linalg.norm(then[i] - then[j], axis=1), np.linalg.norm(now[i] - now[j], axis=1)
            far = before > 60  # nearby pairs are too noisy for a ratio
            if far.sum() < 200:
                continue
            ratios = np.log(after[far] / before[far])
            growth = np.median(ratios)
            if np.median(np.abs(ratios - growth)) > self.max_spread:
                continue
            shift[t] = np.median(at(logs[key], then) + shift[key] - at(logs[t], now)) - growth
            used += 1
        self.used = used / max(1, T - 1)
        self.fast = 0.0
        if self.fast_gap is not None:
            starts, done, nxt, fast = np.unique(keys), [], 1, False
            relaxed = np.zeros_like(shift)
            for t in range(1, T):
                while nxt < len(starts) and starts[nxt] <= t:  # keyframe gaps completed by frame t
                    done.append(starts[nxt] - starts[nxt - 1])
                    nxt += 1
                if len(done) >= 2:
                    gap = np.median(done[-8:])
                    fast = gap <= self.slow_gap if fast else gap < self.fast_gap
                keep = 1 - 1 / self.fast_tau if fast else 1.0
                relaxed[t] = keep * relaxed[t - 1] + shift[t] - shift[t - 1]
                self.fast += fast
            shift = relaxed
            self.fast /= max(1, T - 1)
        return log_depth + torch.from_numpy(shift)[:, None, None]

    def __call__(self, rgb: np.ndarray, log_depth: torch.Tensor, K: np.ndarray | None = None) -> torch.Tensor:
        return self.correct(log_depth, K, *self.track(rgb))


def similarity(a: np.ndarray, b: np.ndarray, keep: float = 0.7) -> tuple[float, float]:
    """Scale s of the similarity a ~ s R b + t (Umeyama), refitted twice on the `keep` share of points that
    fit best. Returns s and the median residual of those points over their median depth in a."""
    inlier = np.ones(len(a), bool)
    for _ in range(3):
        A, B = a[inlier], b[inlier]
        A0, B0 = A - A.mean(0), B - B.mean(0)
        U, S, Vt = np.linalg.svd(A0.T @ B0 / len(A))
        D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
        R = U @ D @ Vt
        s = float(np.trace(np.diag(S) @ D) / (B0**2).sum(1).mean())
        moved = s * b @ R.T + (A.mean(0) - s * R @ B.mean(0))
        resid = np.linalg.norm(a - moved, axis=1)
        inlier = resid <= np.quantile(resid, keep)
    return s, float(np.median(resid[inlier]) / np.median(a[inlier, 2]))
