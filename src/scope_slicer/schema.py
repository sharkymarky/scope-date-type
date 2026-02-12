try:
    from pydantic import BaseModel, Field
except Exception:  # pragma: no cover
    BaseModel = None
    Field = None


if BaseModel is not None:
    class SlicerParams(BaseModel):
        base_slices: int = Field(default=6, ge=3)
        max_slices: int = Field(default=24, ge=3)
        mix: float = Field(default=1.0, ge=0.0, le=1.0)
else:
    class SlicerParams:
        """Lightweight fallback when pydantic is unavailable at runtime."""

        def __init__(self, base_slices: int = 6, max_slices: int = 24, mix: float = 1.0):
            self.base_slices = max(3, int(base_slices))
            self.max_slices = max(3, int(max_slices))
            self.mix = max(0.0, min(1.0, float(mix)))
