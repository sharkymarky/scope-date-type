from __future__ import annotations

import datetime as _dt
import hashlib
import string
import time
from typing import TYPE_CHECKING, Any

import torch
from scope.core.pipelines.interface import Pipeline, Requirements

from .effects import multi_exposure_plate, slit_scan_bands
from .schema import DateTypeConfig, DateTypeMode, ScanOrientation

if TYPE_CHECKING:
    from scope.core.pipelines.base_schema import BasePipelineConfig


def _extract_prompt_text(prompts: Any) -> str:
    if prompts is None:
        return ""
    if isinstance(prompts, (list, tuple)):
        parts: list[str] = []
        for p in prompts:
            if p is None:
                continue
            if isinstance(p, dict):
                t = p.get("text") or p.get("prompt") or ""
            else:
                t = getattr(p, "text", None) or getattr(p, "prompt", None) or ""
            t = str(t).strip()
            if t:
                parts.append(t)
        return " | ".join(parts)
    return str(prompts).strip()


def _count_punct_and_space(s: str) -> tuple[int, int]:
    punct = 0
    spaces = 0
    for ch in s:
        if ch.isspace():
            spaces += 1
        elif ch in string.punctuation:
            punct += 1
    return punct, spaces


def _safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else 0.0


class DateTypePipeline(Pipeline):
    """DATE / TYPE — prompt-driven temporal photography pipeline."""

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return DateTypeConfig

    def __init__(self, **kwargs):
        # Scope passes load params like height/width here.
        try:
            super().__init__(**kwargs)
        except TypeError:
            pass

        # Typing state
        self._prev_text: str = ""
        self._prev_len: int = 0
        self._prev_call_t: float | None = None
        self._last_change_t: float | None = None

        # Date state cache
        self._cached_date_key: str | None = None
        self._cached_date_seed_int: int = 0
        self._cached_date_phase_01: float = 0.0
        self._cached_date_flip_01: float = 0.0
        self._cached_date_bias_01: float = 0.5

    def prepare(self, **kwargs) -> Requirements:
        buffer_len = int(kwargs.get("buffer_len", 60))
        buffer_len = max(8, min(240, buffer_len))
        return Requirements(input_size=buffer_len)

    def _get_date_params(self) -> tuple[str, int, float, float, float]:
        date_key = _dt.date.today().isoformat()
        if date_key == self._cached_date_key:
            return (
                self._cached_date_key,
                self._cached_date_seed_int,
                self._cached_date_phase_01,
                self._cached_date_flip_01,
                self._cached_date_bias_01,
            )

        digest = hashlib.sha256(date_key.encode("utf-8")).digest()
        seed_int = int.from_bytes(digest[:8], "big", signed=False)

        phase_01 = ((seed_int % 1_000_000) / 1_000_000.0)
        flip_01 = digest[8] / 255.0
        bias_01 = digest[9] / 255.0

        self._cached_date_key = date_key
        self._cached_date_seed_int = seed_int
        self._cached_date_phase_01 = phase_01
        self._cached_date_flip_01 = flip_01
        self._cached_date_bias_01 = bias_01

        return date_key, seed_int, phase_01, flip_01, bias_01

    def __call__(self, **kwargs) -> dict:
        video = kwargs.get("video")
        if video is None:
            raise ValueError("DateTypePipeline requires video input")

        frames_u8 = torch.stack([frame.squeeze(0) for frame in video], dim=0)  # (T,H,W,C)
        frames = frames_u8.to(dtype=torch.float32) / 255.0

        prompts = kwargs.get("prompts", None)
        transition = kwargs.get("transition", None)
        text = _extract_prompt_text(prompts)

        now = time.monotonic()
        if self._prev_call_t is None:
            dt = 1.0 / 30.0
        else:
            dt = max(1e-3, now - self._prev_call_t)
        self._prev_call_t = now

        char_len = len(text)
        punct_count, space_count = _count_punct_and_space(text)
        structure_density = _safe_div(float(punct_count + space_count), float(max(1, char_len)))

        delta_len = char_len - self._prev_len
        cadence = abs(delta_len) / dt
        revision = max(0, -delta_len) / dt

        changed = (delta_len != 0) or (text != self._prev_text)
        if changed:
            self._last_change_t = now
        pause = 0.0 if self._last_change_t is None else max(0.0, now - self._last_change_t)

        if transition:
            revision = max(revision, 12.0)

        len_norm = min(1.0, char_len / 240.0)
        cadence_norm = min(1.0, cadence / 24.0)
        revision_norm = min(1.0, revision / 24.0)
        pause_norm = min(1.0, pause / 2.0)
        structure_norm = min(1.0, structure_density / 0.35)

        self._prev_text = text
        self._prev_len = char_len

        enabled = bool(kwargs.get("enabled", True))
        mode = kwargs.get("mode", DateTypeMode.slit_scan)
        orientation = kwargs.get("orientation", ScanOrientation.vertical)

        mix = float(kwargs.get("mix", 1.0))
        smoothing = float(kwargs.get("smoothing", 0.15))
        text_influence = float(kwargs.get("text_influence", 1.0))
        date_influence = float(kwargs.get("date_influence", 0.6))

        exposure_strength = float(kwargs.get("exposure_strength", 0.75))
        memory_decay = float(kwargs.get("memory_decay", 0.55))
        band_count = int(kwargs.get("band_count", 60))

        if not enabled:
            return {"video": frames[-1:].clamp(0, 1)}

        date_key, seed_int, phase_01, flip_01, bias_01 = self._get_date_params()
        date_params = type(
            "DateParams",
            (),
            {"date_key": date_key, "seed_int": seed_int, "phase_01": phase_01, "flip_01": flip_01, "bias_01": bias_01},
        )()

        if mode == DateTypeMode.multi_exposure or str(mode) == str(DateTypeMode.multi_exposure.value):
            out = multi_exposure_plate(
                frames_thwc=frames,
                mix=mix,
                exposure_strength=exposure_strength,
                memory_decay=memory_decay,
                text_influence=text_influence,
                date_influence=date_influence,
                cadence_norm=cadence_norm,
                revision_norm=revision_norm,
                pause_norm=pause_norm,
                date_params=date_params,
            )
        else:
            out = slit_scan_bands(
                frames_thwc=frames,
                band_count=band_count,
                orientation=str(orientation.value) if hasattr(orientation, "value") else str(orientation),
                mix=mix,
                smoothing=smoothing,
                text_influence=text_influence,
                date_influence=date_influence,
                len_norm=len_norm,
                cadence_norm=cadence_norm,
                revision_norm=revision_norm,
                pause_norm=pause_norm,
                structure_norm=structure_norm,
                date_params=date_params,
            )

        return {"video": out.clamp(0, 1)}


