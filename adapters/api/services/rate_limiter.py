from fastapi import Depends
from redis.asyncio import Redis
from datetime import timedelta

from .auth import get_user_from_any_schema
from infrastructure.db.models.user_model import User
from infrastructure.rate_limit import RateLimitService
from infrastructure.redis import get_redis


async def is_allowed(
    redis_conn: Redis = Depends(get_redis),
    usr: User = Depends(get_user_from_any_schema()),
    window: timedelta = timedelta(seconds=60),
    limit: int = 10,
):
    rate_limiter = RateLimitService(client=redis_conn)
    return await rate_limiter.is_allowed(
        key=f"rate_limit:{usr.uuid}", limit=limit, window=window
    )
