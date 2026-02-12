from pydantic import BaseModel, Field


class SlicerParams(BaseModel):
    base_slices: int = Field(default=6, ge=3)
    max_slices: int = Field(default=24, ge=3)
    mix: float = Field(default=1.0, ge=0.0, le=1.0)
