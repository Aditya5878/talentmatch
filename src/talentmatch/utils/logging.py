import logging
import uuid
from contextvars import ContextVar

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

trace_id_var: ContextVar[str] = ContextVar("trace_id", default="")


def get_trace_id() -> str:
    """Return the current trace ID from the context variable.

    Returns:
        The trace ID string, or empty string if not set (e.g. during startup).
    """
    return trace_id_var.get()


class _TraceFilter(logging.Filter):
    """Logging filter that injects trace_id from ContextVar into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "trace_id") or not record.trace_id:
            record.trace_id = trace_id_var.get() or "-"
        return True


def setup_logging() -> None:
    """Configure structured logging for the talentmatch package.

    Sets up a StreamHandler with timestamp, level, logger name, trace ID,
    and message formatting. Adds the _TraceFilter to automatically inject
    trace IDs from the ContextVar. Quiets uvicorn loggers to WARNING level.
    """
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | trace=%(trace_id)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    handler.addFilter(_TraceFilter())

    root = logging.getLogger("talentmatch")
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    root.addFilter(_TraceFilter())

    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logging.getLogger(name).setLevel(logging.WARNING)


class TraceIDMiddleware(BaseHTTPMiddleware):
    """Starlette middleware that propagates trace IDs across requests.

    Extracts X-Trace-ID from the request header (or generates a new one),
    stores it in a ContextVar for logging, and adds it to the response header.
    This enables end-to-end request tracing through all log messages.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        trace_id = request.headers.get("X-Trace-ID") or uuid.uuid4().hex[:16]
        token = trace_id_var.set(trace_id)
        try:
            response = await call_next(request)
            response.headers["X-Trace-ID"] = trace_id
            return response
        finally:
            trace_id_var.reset(token)


def log_event(logger: logging.Logger, level: int, message: str, **extra) -> None:
    """Log a message with the current trace ID automatically attached.

    Args:
        logger: The logger instance to use.
        level: The log level (e.g. logging.INFO, logging.ERROR).
        message: The log message.
        **extra: Additional fields to include in the log record.
    """
    trace_id = get_trace_id()
    logger.log(level, message, extra={"trace_id": trace_id, **extra})
