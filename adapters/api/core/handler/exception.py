from fastapi import Request, status
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from datetime import datetime, timezone, timedelta
from email.utils import formatdate

from domain.exceptions import BaseError

from ...utils import mask_ip
from loggers.log import get_loggers

error_logger = get_loggers("reas.exception")
security_logger = get_loggers("reas.security")

STATUS_CODE_MAP = {
    "PAGINATION_TYPE_ERROR": status.HTTP_400_BAD_REQUEST,
    "DOUBLE_IDENTIFIER_ERROR": status.HTTP_400_BAD_REQUEST,
    "USER_NOT_FOUND": status.HTTP_404_NOT_FOUND,
    "INVALID_CREDENTIALS_ERROR": status.HTTP_401_UNAUTHORIZED,
    "INVALID_FORGET_PASSWORD_TOKEN_ERROR": status.HTTP_401_UNAUTHORIZED,
    "INVALID_VERIFY_EMAIL_TOKEN_ERROR": status.HTTP_401_UNAUTHORIZED,
    "INVALID_AUTHENTICATE_TOKEN_ERROR": status.HTTP_401_UNAUTHORIZED,
    "EMAIL_ALREADY_VERIFIED": status.HTTP_409_CONFLICT,
    "TRANSACTION_UNPAID": status.HTTP_402_PAYMENT_REQUIRED,
    "INVALID_PLAN_REFERENCE": status.HTTP_400_BAD_REQUEST,
    "PLAN_DEACTIVATION_FAILED": status.HTTP_500_INTERNAL_SERVER_ERROR,
    "COLUMN_NOT_FOUND": status.HTTP_400_BAD_REQUEST,
    "INVALID_ENCODING": status.HTTP_400_BAD_REQUEST,
    "DATABASE_ERROR": status.HTTP_500_INTERNAL_SERVER_ERROR,
    "DATABASE_INTEGRITY_ERROR": status.HTTP_500_INTERNAL_SERVER_ERROR,
    "DATABASE_ADAPTER_ERROR": status.HTTP_500_INTERNAL_SERVER_ERROR,
    "DATABASE_UNREACHABLE": status.HTTP_500_INTERNAL_SERVER_ERROR,
    "UNIQUE_CONSTRAINT_VIOLATION": status.HTTP_409_CONFLICT,
    "FOREIGN_KEY_VIOLATION": status.HTTP_409_CONFLICT,
    "CHECK_CONSTRAINT_VIOLATION": status.HTTP_409_CONFLICT,
    "NOT_NULL_CONSTRAINT_VIOLATION": status.HTTP_409_CONFLICT,
    "EMAIL_ALREADY_EXISTS_ERROR": status.HTTP_409_CONFLICT,
    "USER_NAME_ALREADY_EXISTS_ERROR": status.HTTP_409_CONFLICT,
    "FAILED_TO_SEND_EMAIL": status.HTTP_500_INTERNAL_SERVER_ERROR,
    "INSUFFICIENT_ROLE_ERROR": status.HTTP_403_FORBIDDEN,
    "INSUFFICIENT_TIER_ERROR": status.HTTP_402_PAYMENT_REQUIRED,
    "UNSUPPORTED_FILE": status.HTTP_400_BAD_REQUEST,
    "FILE_TOO_LARGE": status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
    "SECURITY_BREACH_POTENTIAL": status.HTTP_400_BAD_REQUEST,
    "DATASET_NAME_ALREADY_EXISTS_ERROR": status.HTTP_409_CONFLICT,
    "DATASET_NOT_FOUND": status.HTTP_404_NOT_FOUND,
    "UNSUPPORTED_DOMAIN_TO_SCRAPE": status.HTTP_400_BAD_REQUEST,
}


def create_retry_after():
    header_value = formatdate(
        (datetime.now(timezone.utc) + timedelta(days=1)).timestamp(), usegmt=True
    )
    return header_value


async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    client_ip = mask_ip(get_remote_address(request))
    error_logger.warning(
        "Rate limit warning for IP: %s, Path: %s", client_ip, request.url.path
    )
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={
            "error_code": "RATE_LIMITED",
            "code": 429,
            "message": "Too many requests, chill out please...",
            "details": {},
        },
        headers={"Retry-After": create_retry_after()},
    )


async def error_handler(request: Request, exc: BaseError):
    """Global exception handler from domain exceptions."""
    status_code = STATUS_CODE_MAP.get(
        exc.error_code, status.HTTP_500_INTERNAL_SERVER_ERROR
    )

    if exc.error_code in [
        "DATABASE_UNREACHABLE",
        "DATABASE_ADAPTER_ERROR",
        "SECURITY_BREACH_POTENTIAL",
    ]:
        if exc.error_code == "SECURITY_BREACH_POTENTIAL":
            detail_dump = exc.details.items()
            security_logger.warning(
                f"Potential security breach at {request.method}/{request.url}: {request.state.request_id} - {request.client.host} `{' - '.join([f'{key} : {value}' for key, value in detail_dump])}`"
            )
        else:
            error_logger.critical(f"Database error: {exc}", exc_info=True)
        return JSONResponse(
            status_code=status_code,
            content={
                "error_code": exc.error_code,
                "message": "An unexpected error occured. Please try again later.",
                "details": {},
            },
        )

    if status_code >= 500:
        error_logger.error(f"Server error: {exc}", exc_info=True)
    else:
        error_logger.info("Client error: %s - %s", exc.error_code, exc.message)

    response_body = JSONResponse(
        status_code=status_code,
        content={
            "error_code": exc.error_code,
            "message": exc.message,
            "details": exc.details,
        },
    )

    if exc.error_code in [
        "INVALID_CREDENTIALS_ERROR",
        "INVALID_AUTHENTICATE_TOKEN_ERROR",
    ]:
        response_body.init_headers({"WWWW-Authenticate": "Bearer"})

    return response_body


async def unexpected_exception_handler(request: Request, exc: Exception):
    """Handle all unexpected exceptions."""
    error_logger.error(
        f"Unexpected error in {request.method} {request.url}: {type(exc).__name__}: {exc}",
        exc_info=True,
        extra={
            "method": request.method,
            "url": str(request.url),
            "request_id": getattr(request.state, "request_id", None),
        },
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error_code": "INTERNAL_ERROR",
            "message": "An unexpected error occured. Please try again later.",
            "details": {},
        },
    )
