from __future__ import annotations

from enum import Enum

from pydantic import Field

from scope.core.pipelines.base_schema import BasePipelineConfig, ModeDefaults, ui_field_config


class DateTypeMode(str, Enum):
    slit_scan = "slit_scan"
    multi_exposure = "multi_exposure"


class ScanOrientation(str, Enum):
    vertical = "vertical"
    horizontal = "horizontal"


class DateTypeConfig(BasePipelineConfig):
    """Configuration for DATE / TYPE (temporal photography) pipeline."""

    pipeline_id = "date-type"
    pipeline_name = "DATE / TYPE"
    pipeline_description = (
        "Prompt-driven temporal photography: slit-scan banding and multi-exposure accumulation "
        "(read-only prompt input; non-semantic typing signals)."
    )

    # Expose prompt box in UI; we read prompts but do not interpret semantics.
    supports_prompts = True

    # Video input required
    modes = {"video": ModeDefaults(default=True)}

    enabled: bool = Field(
        default=True,
        description="Enable DATE / TYPE effect",
        json_schema_extra=ui_field_config(order=1, label="Enabled"),
    )

    mode: DateTypeMode = Field(
        default=DateTypeMode.slit_scan,
        description="Visual mode: slit-scan bands (time as columns) or multi-exposure plate (time as accumulation).",
        json_schema_extra=ui_field_config(order=2, label="Mode"),
    )

    buffer_len: int = Field(
        default=60,
        ge=8,
        le=240,
        description="Number of frames kept in the temporal buffer (load-time; reload pipeline to change).",
        json_schema_extra=ui_field_config(order=3, label="Buffer Length", is_load_param=True),
    )

    band_count: int = Field(
        default=60,
        ge=4,
        le=120,
        description="Number of bands (slit-scan mode).",
        json_schema_extra=ui_field_config(order=4, label="Band Count"),
    )

    orientation: ScanOrientation = Field(
        default=ScanOrientation.vertical,
        description="Band orientation: vertical (columns) or horizontal (rows).",
        json_schema_extra=ui_field_config(order=5, label="Orientation"),
    )

    mix: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Blend between original frame (0) and effect output (1).",
        json_schema_extra=ui_field_config(order=6, label="Mix"),
    )

    smoothing: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Softens seams by applying a 1D blur along the scan axis (0 = hard bands, 1 = very soft).",
        json_schema_extra=ui_field_config(order=7, label="Permeability"),
    )

    text_influence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="How strongly typing dynamics influence structure (0 = static apparatus, 1 = fully responsive).",
        json_schema_extra=ui_field_config(order=8, label="Text Influence"),
    )

    date_influence: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="How strongly today's date influences structure (deterministic per day).",
        json_schema_extra=ui_field_config(order=9, label="Date Influence"),
    )

    memory_decay: float = Field(
        default=0.55,
        ge=0.0,
        le=1.0,
        description="Multi-exposure decay (0 = long persistence, 1 = fast clearing).",
        json_schema_extra=ui_field_config(order=10, label="Memory Decay"),
    )

    exposure_strength: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Strength of multi-exposure accumulation (0 = none, 1 = full plate).",
        json_schema_extra=ui_field_config(order=11, label="Exposure Strength"),
    )
