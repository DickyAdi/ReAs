import time
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from uuid import uuid4

from loggers.log import get_loggers
from ..utils import mask_ip

logger = get_loggers("reas.api")


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        client_host = mask_ip(request.client.host)
        user_agent = request.headers.get("user-agent", "unknown")
        request.state.request_id = str(uuid4())
        logger.info(
            "Incoming request: %s - %s - %s - %s - %s",
            request.state.request_id,
            client_host,
            user_agent,
            request.method,
            request.url.path,
        )

        response = await call_next(request)

        duration = round(time.time() - start_time, 4)

        if response.status_code >= 400:
            logger.warning(
                "Response: %s - %s - %s - %d - %.4fs",
                request.state.request_id,
                request.method,
                request.url.path,
                response.status_code,
                duration,
            )
        else:
            logger.info(
                "Response: %s - %s - %s - %d - %.4fs",
                request.state.request_id,
                request.method,
                request.url.path,
                response.status_code,
                duration,
            )

        return response
