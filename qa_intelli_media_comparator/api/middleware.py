from __future__ import annotations

import time
import uuid
import logging

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

log = logging.getLogger(__name__)


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Attaches a unique request ID and logs timing for every request."""

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = uuid.uuid4().hex[:10]
        request.state.request_id = request_id
        t_start = time.monotonic()

        response: Response = await call_next(request)

        elapsed_ms = int((time.monotonic() - t_start) * 1000)
        response.headers["X-Request-Id"] = request_id
        response.headers["X-Response-Time-Ms"] = str(elapsed_ms)
        log.info(
            "%s %s → %d (%dms) [%s]",
            request.method, request.url.path,
            response.status_code, elapsed_ms, request_id,
        )
        return response
