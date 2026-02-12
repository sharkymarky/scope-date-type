from .schema import SlicerParams

try:
    from scope.types import UsageType
except Exception:  # pragma: no cover
    class UsageType:
        PIPELINE = "pipeline"
        MAIN = "main"


class SlicerPipeline:
    id = "scope_slicer"
    name = "Slicer"
    description = "Slicer effect with prompt-length driven slice count"
    usage = [getattr(UsageType, "PIPELINE", getattr(UsageType, "MAIN", "pipeline"))]
    params_schema = SlicerParams

    def __call__(self, **kwargs):
        params = SlicerParams(
            base_slices=kwargs.get("base_slices", 6),
            max_slices=kwargs.get("max_slices", 24),
            mix=kwargs.get("mix", 1.0),
        )

        prompts = kwargs.get("prompts", [])
        prompt_text = "".join([getattr(p, "text", "") for p in prompts]) if prompts else ""
        slices = max(3, min(params.base_slices + len(prompt_text), params.max_slices))

        video = kwargs.get("video", [])
        if not video:
            return {"video": video}

        try:
            import torch
            from .effects.scope_slicer import apply_slicer
        except Exception:
            # Keep plugin alive even if optional runtime deps are unavailable.
            return {"video": video}

        frames = torch.stack(video, dim=0) if isinstance(video, list) else video
        effected = apply_slicer(frames, slices)
        mixed = frames * (1.0 - params.mix) + effected * params.mix

        return {"video": mixed.clamp(0, 1)}
