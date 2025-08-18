import uuid
import time
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from loggers.log import get_loggers
from .utils import mask_ip

api_logger = get_loggers('reas.api')

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request:Request, call_next):
        start_time = time.time()
        request_id = str(uuid.uuid4())
        client_host = mask_ip(request.client.host)
        user_agent = request.headers.get('user-agent', 'unknown')
        api_logger.info('Incoming request: %s - %s - %s - %s - %s', request_id, client_host, user_agent, request.method, request.url.path)
        
        response = await call_next(request)

        duration = round(time.time() - start_time, 4)

        if response.status_code >= 400:
            api_logger.warning('Response: %s - %s - %s - %d - %.4fs', request_id, request.method, request.url.path, response.status_code, duration)
        else:
            api_logger.info('Response: %s - %s - %s - %d - %.4fs', request_id, request.method, request.url.path, response.status_code, duration)

        return response