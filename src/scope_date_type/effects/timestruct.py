from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class DateParams:
    date_key: str
    seed_int: int
    phase_01: float
    flip_01: float
    bias_01: float


def _band_permutation_indices(S: int, seed_int: int, device: torch.device) -> torch.Tensor:
    """
    Deterministic permutation without RNG state.

    We build a modular permutation: p(i) = (a*i + b) mod S
    where gcd(a, S) == 1.
    """
    if S <= 1:
        return torch.zeros((S,), device=device, dtype=torch.long)

    # Choose an odd 'a' derived from seed; then adjust until coprime to S.
    a = 1 + 2 * (seed_int % max(1, S // 2))
    # Ensure 1 <= a < S
    a = int(a % S) or 1

    def gcd(x: int, y: int) -> int:
        while y:
            x, y = y, x % y
        return x

    while gcd(a, S) != 1:
        a = (a + 2) % S
        if a == 0:
            a = 1

    b = (seed_int // 131) % S

    i = torch.arange(S, device=device, dtype=torch.long)
    return (a * i + b) % S


def _smooth_1d_along_axis(
    frame_thwc: torch.Tensor,
    axis: str,
    strength_0_1: float,
) -> torch.Tensor:
    """Apply a light 1D box blur, blended by strength."""
    if strength_0_1 <= 0:
        return frame_thwc

    # Convert strength to an odd kernel size in [1..31]
    k = int(round(1 + strength_0_1 * 30))
    if k < 3:
        return frame_thwc
    if k % 2 == 0:
        k += 1
    k = min(k, 31)

    x = frame_thwc.permute(0, 3, 1, 2)  # NCHW
    if axis == "x":
        blurred = F.avg_pool2d(x, kernel_size=(1, k), stride=1, padding=(0, k // 2))
    else:
        blurred = F.avg_pool2d(x, kernel_size=(k, 1), stride=1, padding=(k // 2, 0))

    blurred = blurred.permute(0, 2, 3, 1)  # THWC
    return frame_thwc * (1.0 - strength_0_1) + blurred * strength_0_1


def slit_scan_bands(
    frames_thwc: torch.Tensor,
    band_count: int,
    orientation: str,
    mix: float,
    smoothing: float,
    text_influence: float,
    date_influence: float,
    # Typing signals (normalized 0..1)
    len_norm: float,
    cadence_norm: float,
    revision_norm: float,
    pause_norm: float,
    structure_norm: float,
    date_params: DateParams,
) -> torch.Tensor:
    """
    Slit-scan banding: output is composed of bands, each sampled from a different time index.

    frames_thwc: (T, H, W, C), values in [0, 1]
    returns: (1, H, W, C)
    """
    T, H, W, C = frames_thwc.shape
    if T < 2 or H < 1 or W < 1:
        return frames_thwc[-1:].clamp(0, 1)

    device = frames_thwc.device
    newest = frames_thwc[-1:].contiguous()

    # Effective band count: base slider, plus gentle modulation by writing density/structure.
    S_base = int(band_count)
    S_mod = 1.0 + text_influence * 0.30 * (len_norm - 0.5) + text_influence * 0.20 * (structure_norm - 0.5)
    S = int(round(S_base * S_mod))
    S = max(4, min(120, S, T * 4))  # also cap relative to T

    # Base time map across bands: oldest->newest (or flipped daily).
    base_time = torch.linspace(0, T - 1, steps=S, device=device).round().long()
    if date_params.flip_01 < 0.5:
        base_time = (T - 1) - base_time

    # Deterministic band permutation (daily) and phase shift (daily + typing).
    perm = _band_permutation_indices(S, date_params.seed_int, device=device)
    time_per_band = base_time[perm]

    # Phase combines date + typing energy; hesitation damps it.
    phase = (
        date_influence * date_params.phase_01
        + text_influence * (0.55 * cadence_norm + 0.30 * revision_norm - 0.20 * pause_norm)
    ) % 1.0
    phase_shift = int(round(phase * (T - 1)))
    time_per_band = (time_per_band + phase_shift) % T

    if orientation == "horizontal":
        # Each row picks a time index; collapse time along Y axis.
        rows = torch.arange(H, device=device, dtype=torch.long)
        band_id = (rows * S) // H  # (H,)
        t_idx = time_per_band[band_id]  # (H,)

        # Reorder frames to (H, T, W, C), gather along T
        frames_htwc = frames_thwc.permute(1, 0, 2, 3).contiguous()  # (H, T, W, C)
        index = t_idx.view(H, 1, 1, 1).expand(H, 1, W, C)
        out_h1wc = torch.gather(frames_htwc, dim=1, index=index)  # (H, 1, W, C)
        out_hwc = out_h1wc.squeeze(1)  # (H, W, C)
        out = out_hwc.unsqueeze(0)  # (1, H, W, C)

        out = _smooth_1d_along_axis(out, axis="y", strength_0_1=smoothing)
    else:
        # Vertical: each column picks a time index; collapse time along X axis.
        cols = torch.arange(W, device=device, dtype=torch.long)
        band_id = (cols * S) // W  # (W,)
        t_idx = time_per_band[band_id]  # (W,)

        # Reorder frames to (W, T, H, C), gather along T
        frames_wthc = frames_thwc.permute(2, 0, 1, 3).contiguous()  # (W, T, H, C)
        index = t_idx.view(W, 1, 1, 1).expand(W, 1, H, C)
        out_w1hc = torch.gather(frames_wthc, dim=1, index=index)  # (W, 1, H, C)
        out_whc = out_w1hc.squeeze(1)  # (W, H, C)
        out_hwc = out_whc.permute(1, 0, 2).contiguous()  # (H, W, C)
        out = out_hwc.unsqueeze(0)  # (1, H, W, C)

        out = _smooth_1d_along_axis(out, axis="x", strength_0_1=smoothing)

    # Mix between newest and structured output
    mix = float(max(0.0, min(1.0, mix)))
    return (newest * (1.0 - mix) + out * mix).clamp(0, 1)


def multi_exposure_plate(
    frames_thwc: torch.Tensor,
    mix: float,
    exposure_strength: float,
    memory_decay: float,
    text_influence: float,
    date_influence: float,
    # Typing signals (normalized 0..1)
    cadence_norm: float,
    revision_norm: float,
    pause_norm: float,
    date_params: DateParams,
) -> torch.Tensor:
    """
    Marey-inspired multi-exposure: a single frame acts like a plate receiving repeated exposures.

    frames_thwc: (T, H, W, C), values in [0, 1]
    returns: (1, H, W, C)
    """
    T, _H, _W, _C = frames_thwc.shape
    if T < 2:
        return frames_thwc[-1:].clamp(0, 1)

    device = frames_thwc.device
    newest = frames_thwc[-1:].contiguous()

    # Map memory_decay (0..1) to an exponential decay constant k (small = long persistence).
    # Typing energy reduces k (longer persistence). Hesitation increases k (clears plate).
    k_base = 0.7 + 6.0 * float(max(0.0, min(1.0, memory_decay)))
    k_typing = 1.0 - 0.75 * text_influence * cadence_norm + 0.15 * text_influence * revision_norm + 0.35 * text_influence * pause_norm
    k = max(0.15, k_base * max(0.25, k_typing))

    # Daily bias to k (subtle), controlled by date_influence.
    # bias_01 in [0..1] -> [-0.25..+0.25] multiplier
    daily_bias = (date_params.bias_01 - 0.5) * 0.5
    k = k * (1.0 + date_influence * daily_bias)

    # Age-from-newest: 0 for newest, larger for older.
    age = torch.arange(T, device=device, dtype=torch.float32)
    age_from_newest = (T - 1) - age

    weights = torch.exp(-k * age_from_newest)  # newest gets highest
    weights = weights / (weights.sum() + 1e-8)

    plate = (frames_thwc * weights.view(T, 1, 1, 1)).sum(dim=0, keepdim=True)

    exposure_strength = float(max(0.0, min(1.0, exposure_strength)))
    out = newest * (1.0 - exposure_strength) + plate * exposure_strength

    mix = float(max(0.0, min(1.0, mix)))
    return (newest * (1.0 - mix) + out * mix).clamp(0, 1)
