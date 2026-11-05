# EVD — streaming depth for endoscopic video

**Project page:** https://han12580.github.io/EVD/

Inference code for the EVD student: a streaming monocular depth network (DINOv3 ViT encoder + DPT
decoder with a three-tier depth memory). It processes a video frame by frame and keeps one scale for
the whole video: the first frame fixes the (unknown) scale and every later frame continues it.

## Install

```bash
pip install -r requirements.txt
# optional, for --track (scale correction with online point tracks):
pip install git+https://github.com/facebookresearch/co-tracker.git einops
```

## Checkpoints

| file | encoder | parameters |
|---|---|---|
| `evd_s.pt` | DINOv3 ViT-S/16 | 24.5 M |
| `evd_b.pt` | DINOv3 ViT-B/16 | 97.2 M |

Download: [Google Drive](https://drive.google.com/drive/folders/1b09wIH3BYXJj2SPzAw1ve0OgB82oI81v?usp=sharing)
(the encoder weights are fine-tuned from DINOv3 and are subject to the
[DINOv3 License](https://ai.meta.com/resources/models-and-libraries/dinov3-license/)).
Each checkpoint holds the full weights (encoder included) and the
training arguments the model is rebuilt from, so no DINOv3 download is needed.

## Run

```bash
# a video file or a directory of frames
python predict.py --checkpoint evd_s.pt --input video.mp4 --out out/ --png
```

Output: `out/depth/000000.npy`, ... — float32 depth per frame at the processing resolution, all frames
in one shared, unknown scale (multiply by one constant to get metric depth). Frames whose sides are not
multiples of 16 are resized to a long side of 640 (`--long-side` overrides).

### Scale correction with point tracks (optional)

```bash
python predict.py --checkpoint evd_s.pt --input video.mp4 --out out/ \
    --track --intrinsics FX FY CX CY --cotracker checkpoints/scaled_online.pth
```

CoTracker3 (online, forward only) follows a grid of points between keyframes; each frame's tracked
points are lifted to 3D with the predicted depth and the camera intrinsics (`--intrinsics`, of the input
frames), and a similarity fit to the keyframe sets that frame's scale. Only one constant per frame is
changed, never the shape. When the camera moves fast (keyframes closer than 16 frames), the correction
relaxes to the network's own scale with a time constant of 50 frames. `scaled_online.pth` is the
CoTracker3 online checkpoint from the CoTracker repository.

## Python

```python
import torch
from evd import load_model, normalize

model, _ = load_model("evd_s.pt", "cuda")
state = None
for clip in clips:                      # (T, H, W, 3) uint8 RGB, H and W multiples of 16
    video = normalize(torch.from_numpy(clip).cuda())[None]
    with torch.no_grad():
        log_depth, state = model(video, state)   # pass `state` back to continue the same stream
```

## Licences

`evd/dinov3/` is the DINOv3 ViT implementation from Hugging Face Transformers (Apache-2.0, header
kept). The encoder weights inside the checkpoints are fine-tuned from Meta's DINOv3 weights and remain
subject to the [DINOv3 License](https://ai.meta.com/resources/models-and-libraries/dinov3-license/).
