from scope.core.plugins.hookspecs import hookimpl


@hookimpl
def register_pipelines(register):
    # Lazy import keeps Scope startup fast and avoids importing torch unless needed.
    from .pipeline import DateTypePipeline

    register(DateTypePipeline)
