from redis.asyncio import Redis
from time import time
from datetime import timedelta

from domain.rate_limit import RateLimitInterface
from domain.exceptions import RequestRateLimitedError


class RateLimitService(RateLimitInterface):
    def __init__(self, client: Redis):
        self._client = client

    async def is_allowed(
        self, key: str, limit: int, window: timedelta = timedelta(seconds=60)
    ) -> bool:
        now = time()
        window_start = now - window.total_seconds()

        async with self._client.pipeline(transaction=True) as pipe:
            await pipe.zremrangebyscore(key, 0, window_start)
            await pipe.zadd(key, {str(now): now})
            await pipe.zcard(key)
            await pipe.expire(key, int(window.total_seconds()))
            results = await pipe.execute()

        # return results[2] <= limit
        if (
            not results[2] <= limit
        ):  # if given request surpass limit threshold, will raise error
            raise RequestRateLimitedError(limit=limit, window=int(window.total_seconds))
        return True
