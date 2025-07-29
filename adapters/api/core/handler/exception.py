from fastapi import Request, status
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from ...utils import mask_ip
from loggers.log import get_loggers

rate_limit_logger = get_loggers('reas.exception')

async def rate_limit_handler(request:Request, exc:RateLimitExceeded):
    client_ip = mask_ip(get_remote_address(request))
    rate_limit_logger.warning('Rate limit warning for IP: %s, Path: %s', client_ip, request.url.path)
    return JSONResponse(status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        content={
                            'status' : 'error',
                            'code' : 429,
                            'message' : 'Too many requests, chill out please...'
                        })