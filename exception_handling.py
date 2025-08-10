from functools import wraps
from typing import Callable, Any, Optional
from logging import Logger
from logging import getLogger

class PipelineError(Exception):
    """Custom exception for pipeline-related failures."""

def log_exceptions(logger: Optional[Logger] = None):
    logger = logger or getLogger("pipeline")

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.exception(f"Exception in {func.__name__}: {e}")
                raise PipelineError(str(e)) from e
        return wrapper
    return decorator
