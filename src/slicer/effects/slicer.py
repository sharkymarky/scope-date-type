import math

import torch
import torch.nn.functional as F


def apply_slicer(frames: torch.Tensor, slices: int) -> torch.Tensor:
    """Apply a rotational slicer remap using grid_sample."""
    b, _, h, w = frames.shape
    device = frames.device
    dtype = frames.dtype

    ys = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")

    r = torch.sqrt(xx * xx + yy * yy)
    theta = torch.atan2(yy, xx)

    wedge = (2.0 * math.pi) / max(1, slices)
    theta_fold = torch.remainder(theta + math.pi, wedge)
    theta_mirror = torch.where(theta_fold > (wedge * 0.5), wedge - theta_fold, theta_fold)

    x_map = r * torch.cos(theta_mirror)
    y_map = r * torch.sin(theta_mirror)

    grid = torch.stack((x_map, y_map), dim=-1).unsqueeze(0).expand(b, -1, -1, -1)
    return F.grid_sample(frames, grid, mode="bilinear", padding_mode="border", align_corners=True)
