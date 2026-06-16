"""
video_augmentations_v2.py
=========================
Enhanced augmentation pipeline for large-scale LSC/ASL pretraining.

Design philosophy
-----------------
The goal is NOT photorealism — it is dataset-agnostic feature learning.
Every augmentation should break a cue that is dataset-specific but
irrelevant to the sign being performed:
  - Background colour/texture  → BackgroundRemovalAndReplace
  - Recording speed            → TemporalScale, RandomFrameSkip
  - Camera distance / framing  → SpatialJitter
  - Lighting / skin tone shift → ColorJitter, RandomGrayscale
  - Camera shake               → RandomAffinePerFrame
  - Video codec noise          → GaussianNoise, RandomBlockDropout
  - Mirror conventions         → RandomHorizontalFlip (signing is *not*
                                  laterally symmetric, so use sparingly and
                                  only during pretraining, disable for
                                  medical fine-tuning)

Tensor convention everywhere: (C, T, H, W), float32, values in [0, 1].
"""

from __future__ import annotations

import glob
import logging
import os
import random
import time
import math
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.io import read_image


# ===========================================================================
#  Temporal augmentations
# ===========================================================================

class RandomFrameSkip:
    """
    Randomly drops intermediate frames.
    Always keeps first and last frame so temporal endpoints are stable.

    skip_prob: per-frame probability of being dropped (0 = keep all).
    """
    def __init__(self, skip_prob: float = 0.2):
        self.skip_prob = skip_prob

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        C, T, H, W = video.shape
        if T <= 3:
            return video

        indices = [0]
        for i in range(1, T - 1):
            if random.random() > self.skip_prob:
                indices.append(i)
        indices.append(T - 1)

        return video[:, indices, :, :]


class TemporalScale:
    """
    Resamples the video along the time axis.
    scale < 1 → faster (fewer frames), scale > 1 → slower (more frames).
    Uses nearest-neighbour to avoid temporal blur.
    """
    def __init__(self, scale_range: Tuple[float, float] = (0.7, 1.4)):
        self.scale_range = scale_range

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        C, T, H, W = video.shape
        scale = random.uniform(*self.scale_range)
        new_T = max(1, int(T * scale))
        if new_T == T:
            return video

        video_b = video.unsqueeze(0)   # (1, C, T, H, W)
        scaled = F.interpolate(video_b, size=(new_T, H, W), mode='nearest')
        return scaled.squeeze(0)


class TemporalReverse:
    """
    Reverses the frame order with probability p.
    Signs played backwards are meaningless — forces the model not to rely
    on absolute temporal direction as a shortcut.
    """
    def __init__(self, p: float = 0.15):
        self.p = p

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            return video.flip(dims=[1])   # flip along T
        return video


class RandomTemporalCrop:
    """
    Crops a contiguous subsequence of at least min_keep_ratio of the video.
    Simulates partial visibility / trimming errors.
    """
    def __init__(self, min_keep_ratio: float = 0.7):
        self.min_keep_ratio = min_keep_ratio

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        C, T, H, W = video.shape
        if T <= 4:
            return video
        min_frames = max(2, int(T * self.min_keep_ratio))
        keep = random.randint(min_frames, T)
        start = random.randint(0, T - keep)
        return video[:, start:start + keep, :, :]


# ===========================================================================
#  Spatial / photometric augmentations (applied per-frame)
# ===========================================================================

class RandomHorizontalFlip:
    """
    Horizontally mirrors the entire video.
    WARNING: disable during medical fine-tuning — handedness matters in LSC.
    For pretraining on heterogeneous data it helps break recording-side bias.
    """
    def __init__(self, p: float = 0.3):
        self.p = p

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            return video.flip(dims=[3])   # flip along W
        return video


class ColorJitter:
    """
    Applies random brightness / contrast / saturation / hue shift
    consistently across all frames.
    Breaks lighting and camera white-balance differences between datasets.
    """
    def __init__(
        self,
        brightness: float = 0.4,
        contrast: float = 0.4,
        saturation: float = 0.4,
        hue: float = 0.08,
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def _sample_factor(self, magnitude: float) -> float:
        return random.uniform(max(0.0, 1 - magnitude), 1 + magnitude)

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        C, T, H, W = video.shape
        v = video.permute(1, 0, 2, 3)  # (T, C, H, W)

        # Each adjust_* returns a new tensor — we reassign v to free the previous one
        # immediately rather than holding all 4 alive at once
        if self.brightness:
            v = TF.adjust_brightness(v, self._sample_factor(self.brightness))
        if self.contrast:
            v = TF.adjust_contrast(v, self._sample_factor(self.contrast))
        if self.saturation:
            v = TF.adjust_saturation(v, self._sample_factor(self.saturation))
        if self.hue:
            v = TF.adjust_hue(v, random.uniform(-self.hue, self.hue))

        return v.permute(1, 0, 2, 3)


class RandomGrayscale:
    """
    Converts the full clip to grayscale with probability p.
    Forces colour-invariant features — useful because some datasets are
    recorded with different colour profiles.
    """
    def __init__(self, p: float = 0.1):
        self.p = p

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            # Mean across channels, then broadcast back to 3 channels
            gray = video.mean(dim=0, keepdim=True).expand_as(video)
            return gray
        return video


class GaussianNoise:
    """
    Same behaviour, but reuses a pre-allocated buffer instead of calling
    torch.randn_like() on every forward pass. Eliminates the malloc entirely
    after the first call.
    """
    def __init__(self, std_range: Tuple[float, float] = (0.0, 0.04)):
        self.std_range = std_range
        self._buf: Optional[torch.Tensor] = None

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        std = random.uniform(*self.std_range)
        if std == 0.0:
            return video
        # Reallocate only when video shape changes (e.g. after TemporalScale)
        if self._buf is None or self._buf.shape != video.shape:
            self._buf = torch.empty_like(video)
        self._buf.normal_(0.0, std)          # fill in-place, no malloc
        return video.add(self._buf).clamp_(0.0, 1.0)


class RandomBlockDropout:
    """
    Zeros out a random rectangular region in all frames (simulates occlusion,
    logo overlays, subtitles that partially occlude the signer).
    """
    def __init__(self, p: float = 0.15, max_ratio: float = 0.2):
        self.p = p
        self.max_ratio = max_ratio

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return video
        C, T, H, W = video.shape
        bh = random.randint(1, max(1, int(H * self.max_ratio)))
        bw = random.randint(1, max(1, int(W * self.max_ratio)))
        y0 = random.randint(0, H - bh)
        x0 = random.randint(0, W - bw)
        video = video.clone()
        video[:, :, y0:y0 + bh, x0:x0 + bw] = 0.0
        return video


class RandomAffinePerFrame:
    """
    Applies a random walk of slight rotations and translations (shifting up/down/left/right) independently to each frame
    Simulates camera shake and slight tracking instabilitie
    """
    def __init__(
        self,
        max_angle: float = 5.0,
        max_translate: float = 0.05,
        p: float = 0.3,
    ):
        self.max_angle = max_angle
        self.max_translate = max_translate
        self.p = p

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return video

        C, T, H, W = video.shape

        # Pre-compute all per-frame parameters via random walk
        angles  = np.empty(T)
        txs     = np.empty(T)
        tys     = np.empty(T)
        angle   = random.uniform(-self.max_angle, self.max_angle)
        tx      = random.uniform(-self.max_translate, self.max_translate)
        ty      = random.uniform(-self.max_translate, self.max_translate)
        for t in range(T):
            angle    += random.gauss(0, self.max_angle * 0.1)
            tx       += random.gauss(0, self.max_translate * 0.1)
            ty       += random.gauss(0, self.max_translate * 0.1)
            angles[t] = float(np.clip(angle, -self.max_angle, self.max_angle))
            txs[t]    = int(np.clip(tx * W, -W * self.max_translate, W * self.max_translate))
            tys[t]    = int(np.clip(ty * H, -H * self.max_translate, H * self.max_translate))

        # Pre-allocate output — no list, no torch.stack peak
        out = torch.empty_like(video)
        for t in range(T):
            out[:, t, :, :] = TF.affine(
                video[:, t, :, :],
                angle=angles[t],
                translate=[txs[t], tys[t]],
                scale=1.0,
                shear=0.0,
            )
        return out


class SpatialJitter:
    """
    Randomly crops and resizes each clip — simulates different zoom levels
    and camera distances across datasets.
    """
    def __init__(self, scale_range: Tuple[float, float] = (0.8, 1.0)):
        self.scale_range = scale_range

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        C, T, H, W = video.shape
        scale = random.uniform(*self.scale_range)
        crop_h = int(H * scale)
        crop_w = int(W * scale)
        top  = random.randint(0, H - crop_h)
        left = random.randint(0, W - crop_w)

        video = video[:, :, top:top + crop_h, left:left + crop_w]
        # Resize back to original resolution
        video_b = video.unsqueeze(0)
        video_b = F.interpolate(video_b, size=(T, H, W), mode='nearest')
        return video_b.squeeze(0)


# ===========================================================================
#  Background removal + replacement
# ===========================================================================

class BackgroundRemovalAndReplace:
    def __init__(
        self,
        backgrounds: List[torch.Tensor],
        p: float = 0.8,
        chroma_threshold: float = 0.15,
        blur_kernel: Optional[int] = 3,      # set to None to skip blur entirely
        target_size: Optional[Tuple[int, int]] = None,
    ):
        self.p = p
        self.chroma_threshold = chroma_threshold
        self.blur_kernel = blur_kernel

        # Pre‑resize backgrounds if target size is known
        if target_size is not None:
            self.backgrounds = [TF.resize(bg, target_size) for bg in backgrounds]
        else:
            self.backgrounds = backgrounds
        self._cached_size = target_size

    def _chroma_mask(self, video: torch.Tensor) -> torch.Tensor:
        R = video[0:1]    # shape (1, T, H, W)
        G = video[1:2]
        B = video[2:3]
        bg_mask = (G > R + self.chroma_threshold) & (G > B + self.chroma_threshold)
        return ~bg_mask          # bool, (1, T, H, W)

    @staticmethod
    def _smooth_mask_box(mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
        # Expects mask (1, T, H, W).  Convert to float for pooling.
        m = mask.float().permute(1, 0, 2, 3)      # (T, 1, H, W)
        pad = kernel_size // 2
        m = F.pad(m, (pad, pad, pad, pad), mode='reflect')
        blurred = F.avg_pool2d(m, kernel_size, stride=1)
        return blurred.permute(1, 0, 2, 3).clamp(0.0, 1.0)

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p or not self.backgrounds:
            return video

        C, T, H, W = video.shape
        bg_image = random.choice(self.backgrounds)

        # Lazy one‑time resize if size changed (e.g. after SpatialJitter)
        if (H, W) != self._cached_size:
            bg_resized = TF.resize(bg_image, (H, W)).unsqueeze(1)   # (C, 1, H, W)
            self._cached_size = (H, W)
        else:
            # bg_image is already (C, H, W) – add T dim
            bg_resized = bg_image.unsqueeze(1) if bg_image.dim() == 3 else bg_image

        alpha = self._chroma_mask(video)          # bool, (1, T, H, W)
        if self.blur_kernel:
            alpha = self._smooth_mask_box(alpha, self.blur_kernel)

        # alpha * foreground + (1 - alpha) * background
        composited = alpha * video + (1.0 - alpha) * bg_resized
        return composited.clamp(0.0, 1.0)
    
    
# ===========================================================================
#  Legacy chroma-key (kept for backward compatibility)
# ===========================================================================

class ChromaKeyBackgroundChange:
    """
    Simple chroma-key replacement (original implementation, kept for compat).
    Prefer BackgroundRemovalAndReplace for pretraining.
    """
    def __init__(self, backgrounds: List[torch.Tensor], green_threshold: float = 0.15):
        self.backgrounds = backgrounds
        self.green_threshold = green_threshold

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        C, T, H, W = video.shape
        R, G, B = video[0], video[1], video[2]
        mask = (G > R + self.green_threshold) & (G > B + self.green_threshold)
        mask = mask.unsqueeze(0).expand_as(video)
        bg_image = random.choice(self.backgrounds)
        bg_image = TF.resize(bg_image, size=(H, W))
        bg_video = bg_image.unsqueeze(1).expand_as(video)
        return torch.where(mask, bg_video, video)


# ===========================================================================
#  Composition helpers
# ===========================================================================

# Replace VideoAugmentationPipeline with this version

class VideoAugmentationPipeline:
    """
    Composes a list of augmentations in order, optionally timing each step.
    """
    def __init__(self, augmentations: list, profile: bool = False):
        self.augmentations = augmentations
        self.profile = profile
        self._timings: dict[str, list[float]] = {}   # name → list of elapsed seconds
        
        # Track video size metrics
        self.total_frames = 0
        self.total_videos = 0

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        if self.profile:
            self.total_videos += 1
            # Extract number of frames (T) from shape (C, T, H, W)
            self.total_frames += video.shape[1]

        for aug in self.augmentations:
            if self.profile:
                name = type(aug).__name__
                t0 = time.perf_counter()
                video = aug(video)
                elapsed = time.perf_counter() - t0
                self._timings.setdefault(name, []).append(elapsed)
                logger.debug(f"[profile] {name}: {elapsed * 1000:.1f} ms")
            else:
                video = aug(video)
        return video

    def print_timing_report(self) -> None:
        """Prints a summary table of cumulative and per-call timings."""
        if not self._timings:
            print("No profiling data — construct the pipeline with profile=True.")
            return

        total_all = sum(sum(v) for v in self._timings.values())
        rows = []
        for name, times in self._timings.items():
            total = sum(times)
            rows.append((name, len(times), total * 1000, total / max(total_all, 1e-9) * 100))

        rows.sort(key=lambda r: r[2], reverse=True)   # slowest first

        header = f"{'Transform':<35} {'Calls':>6} {'Total (ms)':>12} {'Share':>8}"
        print("\n" + "─" * len(header))
        print(header)
        print("─" * len(header))
        for name, calls, ms, pct in rows:
            print(f"{name:<35} {calls:>6} {ms:>12.1f} {pct:>7.1f}%")
        print("─" * len(header))
        print(f"{'TOTAL TIME':<35} {'':>6} {total_all * 1000:>12.1f} {'100.0%':>8}")
        print("─" * len(header))
        
        # Print size metrics and speed comparisons
        print(f"Videos Processed:  {self.total_videos}")
        print(f"Frames Processed:  {self.total_frames}")
        if self.total_frames > 0:
            ms_per_frame = (total_all * 1000) / self.total_frames
            fps_processing = self.total_frames / total_all
            print(f"Avg Speed:         {ms_per_frame:.2f} ms/frame ({fps_processing:.1f} FPS)")
        print("─" * len(header) + "\n")
        
        


def build_pretraining_pipeline(
    backgrounds: List[torch.Tensor],
    profile: bool = False,     

) -> VideoAugmentationPipeline:
    """
    Returns the recommended heavy augmentation pipeline for large-scale
    pretraining on mixed LSC + How2Sign data.

    Background removal is applied FIRST so subsequent spatial augmentations
    operate on the composited image.
    """
    return VideoAugmentationPipeline([
        # 1. Background agnosticism (most important for cross-dataset generalisation)
        BackgroundRemovalAndReplace(
            backgrounds=backgrounds,
            p=0.85,
            target_size=(224, 224), 
            
        ),

        # 2. Temporal augmentations — break recording speed / FPS differences
        RandomTemporalCrop(min_keep_ratio=0.75),
        TemporalScale(scale_range=(0.75, 1.35)),
        RandomFrameSkip(skip_prob=0.15),
        TemporalReverse(p=0.1),

        # 3. Spatial jitter — break camera distance / framing differences
        SpatialJitter(scale_range=(0.82, 1.0)),
        RandomAffinePerFrame(max_angle=4.0, max_translate=0.04, p=0.4),

        # 4. Photometric augmentations — break lighting / camera response
        # ColorJitter(brightness=0.35, contrast=0.35, saturation=0.3, hue=0.06),
        RandomGrayscale(p=0.1),
        # GaussianNoise(std_range=(0.0, 0.035)),

        # 5. Occlusion / corruption
        RandomBlockDropout(p=0.15, max_ratio=0.18),

        # 6. Horizontal flip — use during pretraining only; disable for LSC fine-tuning
        RandomHorizontalFlip(p=0.25),
    ], profile=profile)


def build_finetuning_pipeline(
    backgrounds: List[torch.Tensor],
) -> VideoAugmentationPipeline:
    """
    Conservative pipeline for medical LSC fine-tuning on small datasets.

    Key differences from pretraining:
    - No RandomHorizontalFlip  — handedness is semantically meaningful in LSC
    - No TemporalReverse       — temporal direction carries meaning in signs
    - No ColorJitter           — too expensive; color is a weak cue anyway
    - No GaussianNoise         — too expensive for marginal benefit at fine-tune scale
    - No RandomTemporalCrop    — small datasets can't afford to lose sign boundaries
    - Lower p on background replace — reduces training instability on small N
    - Narrower TemporalScale   — preserves signing speed as a feature
    - Smaller affine params    — avoids distorting hand shape, which is discriminative
    - Smaller BlockDropout     — simulates partial occlusion without destroying key frames
    """
    return VideoAugmentationPipeline([
        # 1. Background agnosticism — still your highest-ROI augmentation,
        #    but lower p reduces instability on small datasets.
        BackgroundRemovalAndReplace(
            backgrounds=backgrounds,
            p=0.75,
            target_size=(224, 224),
        ),

        # 2. Temporal — very conservative; preserve signing speed and direction.
        TemporalScale(scale_range=(0.9, 1.1)),
        RandomFrameSkip(skip_prob=0.05),
        # TemporalReverse intentionally omitted
        # RandomTemporalCrop intentionally omitted

        # 3. Spatial — simulate camera distance and minor shake only.
        #    max_angle=2.0 avoids rotating hands out of their canonical orientation.
        SpatialJitter(scale_range=(0.92, 1.0)),
        RandomAffinePerFrame(max_angle=2.0, max_translate=0.02, p=0.2),
        # RandomHorizontalFlip intentionally omitted

        # 4. Occlusion — lighter than pretraining; simulates clothing/props overlap.
        RandomBlockDropout(p=0.10, max_ratio=0.12),

        # ColorJitter intentionally omitted — too slow, sign recognition is
        # primarily shape-driven, not color-driven.
        # GaussianNoise intentionally omitted — same reasoning.
    ])


# ===========================================================================
#  Utilities
# ===========================================================================

def load_backgrounds_from_folder(folder_path: str) -> List[torch.Tensor]:
    """Loads all jpg/png images from a folder into normalised float tensors."""
    bg_tensors = []
    image_paths = glob.glob(os.path.join(folder_path, "*.[jp][pn]g"))
    for path in image_paths:
        img = read_image(path).float() / 255.0
        bg_tensors.append(img)
    return bg_tensors


# ===========================================================================
#  NEW: Ablation-motivated augmentations (v3)
#  Added based on ablation results: spatial > temporal > background > occlusion
# ===========================================================================

class TemporalAccelDecel:
    """
    Applies a non-uniform time warp: the first half of the video is played
    at a different speed than the second half. Simulates natural signing
    rhythm variation — signers often accelerate into a sign and decelerate at
    the boundary, or vice versa.

    Directly motivated by the ablation: temporal was the second most impactful
    group. The existing TemporalScale applies a *uniform* speed change. This
    class adds *non-uniform* warping that TemporalScale cannot produce,
    exposing the model to a wider range of intra-clip speed profiles.

    accel_range: (min, max) speed ratio for each segment (chosen independently).
    p:           probability of applying this transform.
    """
    def __init__(
        self,
        accel_range: Tuple[float, float] = (0.75, 1.4),
        p: float = 0.4,
    ):
        self.accel_range = accel_range
        self.p = p

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return video
        C, T, H, W = video.shape
        if T < 4:
            return video

        split  = random.randint(T // 4, 3 * T // 4)   # split between 25%–75%
        scale1 = random.uniform(*self.accel_range)
        scale2 = random.uniform(*self.accel_range)

        new_T1 = max(1, int(split * scale1))
        new_T2 = max(1, int((T - split) * scale2))

        seg1 = F.interpolate(video[:, :split, :, :].unsqueeze(0),   size=(new_T1, H, W), mode='nearest').squeeze(0)
        seg2 = F.interpolate(video[:, split:, :, :].unsqueeze(0),   size=(new_T2, H, W), mode='nearest').squeeze(0)
        return torch.cat([seg1, seg2], dim=1)


class MultiScaleSpatialJitter:
    """
    Applies TWO sequential spatial crops at different scales, producing a
    richer distribution of signer positions and sizes than a single
    SpatialJitter. Directly motivated by spatial being the #1 group in the
    ablation — we want maximum diversity in how the signer is framed.

    The second crop is applied with probability p (so ~50% of the time only
    one crop fires, keeping the augmentation from being too destructive on a
    small dataset).

    scale_range: (min, max) for each individual crop step.
    p:           probability of applying the second crop step.
    """
    def __init__(
        self,
        scale_range: Tuple[float, float] = (0.78, 1.0),
        p_second_crop: float = 0.5,
    ):
        self.scale_range = scale_range
        self.p_second_crop = p_second_crop

    def _one_crop(self, video: torch.Tensor, H: int, W: int) -> torch.Tensor:
        C, T, ch, cw = video.shape
        scale  = random.uniform(*self.scale_range)
        crop_h = max(1, int(ch * scale))
        crop_w = max(1, int(cw * scale))
        top    = random.randint(0, ch - crop_h)
        left   = random.randint(0, cw - crop_w)
        cropped = video[:, :, top:top + crop_h, left:left + crop_w]
        return F.interpolate(cropped.unsqueeze(0), size=(T, H, W), mode='nearest').squeeze(0)

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        C, T, H, W = video.shape
        video = self._one_crop(video, H, W)
        if random.random() < self.p_second_crop:
            video = self._one_crop(video, H, W)
        return video


class RandomSignerScale:
    """
    Randomly resizes the signer *within* the frame by scaling the video
    content down and padding the remainder with black. Unlike SpatialJitter,
    which can only zoom IN (crop inward), this class can zoom OUT — the signer
    appears smaller, surrounded by empty space. Expands the spatial
    distribution well beyond what SpatialJitter alone produces.

    Directly motivated by the ablation: spatial was the #1 group.

    scale_range: fraction of the frame the signer content occupies.
                 (0.6, 1.0) means the signer can appear at 60%–100% of
                 original size, placed at a random position in the frame.
    p:           probability of applying.
    """
    def __init__(
        self,
        scale_range: Tuple[float, float] = (0.6, 1.0),
        p: float = 0.35,
    ):
        self.scale_range = scale_range
        self.p = p

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return video
        C, T, H, W = video.shape
        scale  = random.uniform(*self.scale_range)
        new_h  = max(1, int(H * scale))
        new_w  = max(1, int(W * scale))

        small = F.interpolate(
            video.unsqueeze(0), size=(T, new_h, new_w), mode='nearest'
        ).squeeze(0)

        pad_top  = random.randint(0, H - new_h)
        pad_left = random.randint(0, W - new_w)

        canvas = torch.zeros_like(video)
        canvas[:, :, pad_top:pad_top + new_h, pad_left:pad_left + new_w] = small
        return canvas