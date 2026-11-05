"""Streaming depth for an endoscopic video with an EVD student checkpoint.

    python predict.py --checkpoint evd_s.pt --input video.mp4 --out out/
    python predict.py --checkpoint evd_b.pt --input frames/ --out out/ --png
    python predict.py --checkpoint evd_s.pt --input video.mp4 --out out/ \
        --track --intrinsics 600 600 320 256 --cotracker checkpoints/scaled_online.pth

Writes one depth map per frame (float32 .npy, at the processing resolution). The whole video shares
one unknown scale: the first frame fixes it and every later frame continues it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from evd.model import load_model, normalize

IMAGES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def read_frames(path: Path):
    """Yield RGB uint8 frames from a video file or from a directory of images (sorted by name)."""
    if path.is_dir():
        for file in sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGES):
            yield cv2.cvtColor(cv2.imread(str(file)), cv2.COLOR_BGR2RGB)
        return
    video = cv2.VideoCapture(str(path))
    while True:
        ok, frame = video.read()
        if not ok:
            break
        yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    video.release()


def working_size(h: int, w: int, long_side: int | None) -> tuple[int, int]:
    """Both sides must be multiples of 16. Sizes that are not (or any size when --long-side is given)
    are resized to that long side (default 640, as the 512x640 training frames)."""
    if long_side is None and h % 16 == 0 and w % 16 == 0:
        return h, w
    side = long_side or 640
    return tuple(max(16, round(s * side / max(h, w) / 16) * 16) for s in (h, w))


def colorize(depth: np.ndarray) -> np.ndarray:
    """Inverse depth, 2nd-98th percentile, as an 8-bit colour image (near = bright)."""
    disp = 1.0 / np.maximum(depth, 1e-6)
    lo, hi = np.percentile(disp, (2, 98))
    gray = np.clip((disp - lo) / max(hi - lo, 1e-6), 0, 1)
    return cv2.applyColorMap((gray * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="EVD student checkpoint (.pt)")
    parser.add_argument("--input", required=True, type=Path, help="video file or directory of frames")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--long-side", type=int, help="resize to this long side (multiple of 16)")
    parser.add_argument("--chunk", type=int, default=16, help="frames encoded per call; does not change the result")
    parser.add_argument("--fp16", action="store_true", help="half-precision autocast (the released models are fp32)")
    parser.add_argument("--png", action="store_true", help="also write colourised depth images")
    parser.add_argument("--track", action="store_true", help="correct each frame's scale with point tracks")
    parser.add_argument("--intrinsics", type=float, nargs=4, metavar=("FX", "FY", "CX", "CY"),
                        help="pinhole intrinsics of the INPUT frames, needed by --track")
    parser.add_argument("--cotracker", default="checkpoints/scaled_online.pth",
                        help="CoTracker3 online checkpoint, needed by --track")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.track and args.intrinsics is None:
        parser.error("--track needs --intrinsics")

    model, _ = load_model(args.checkpoint, args.device)
    (args.out / "depth").mkdir(parents=True, exist_ok=True)
    if args.png:
        (args.out / "png").mkdir(exist_ok=True)

    size, state, chunk, logs, frames, scale = None, None, [], [], [], None
    index = 0

    def run(batch: list[np.ndarray]) -> None:
        nonlocal state, index
        video = normalize(torch.from_numpy(np.stack(batch)).to(args.device))
        if video.shape[-2:] != size:
            video = F.interpolate(video, size=size, mode="bilinear", align_corners=False)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=args.fp16):
            log_depth, state = model(video[None], state)
        for frame in log_depth[0].float().cpu():
            if args.track:
                logs.append(frame)  # written after the tracker has seen the whole video
            else:
                save(index, frame.exp().numpy())
            index += 1

    def save(i: int, depth: np.ndarray) -> None:
        np.save(args.out / "depth" / f"{i:06d}.npy", depth.astype(np.float32))
        if args.png:
            cv2.imwrite(str(args.out / "png" / f"{i:06d}.png"), colorize(depth))

    for rgb in read_frames(args.input):
        if size is None:
            h, w = rgb.shape[:2]
            size = working_size(h, w, args.long_side)
            scale = (size[1] / w, size[0] / h)
        if args.track:
            frames.append(cv2.resize(rgb, (size[1], size[0]), interpolation=cv2.INTER_AREA)
                          if rgb.shape[:2] != size else rgb)
        chunk.append(rgb)
        if len(chunk) == args.chunk:
            run(chunk)
            chunk = []
    if chunk:
        run(chunk)
    if index == 0:
        raise SystemExit(f"no frames read from {args.input}")

    if args.track:
        from evd.scale_track import ScaleTracker

        fx, fy, cx, cy = args.intrinsics
        sx, sy = scale  # intrinsics at the processing size; pixel centres: u_out + .5 = s (u_in + .5)
        K = np.array([[fx * sx, 0, (cx + 0.5) * sx - 0.5], [0, fy * sy, (cy + 0.5) * sy - 0.5], [0, 0, 1]])
        tracker = ScaleTracker(args.device, mode="3d", fast_gap=16, slow_gap=32, fast_tau=50,
                               checkpoint=args.cotracker)
        corrected = tracker(np.stack(frames), torch.stack(logs), K)
        for i, frame in enumerate(corrected):
            save(i, frame.exp().numpy())
    print(f"{index} frames at {size[0]}x{size[1]} -> {args.out / 'depth'}")


if __name__ == "__main__":
    main()
