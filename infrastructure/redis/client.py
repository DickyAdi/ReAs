# from redis import Redis
from redis.asyncio import Redis, ConnectionPool
from redis.exceptions import ConnectionError
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from asyncio import Semaphore

from config.settings import settings


class RedisConnectionPool:
    def __init__(self):
        self.pool = None
        self._semaphore = None

    async def initialize(self):
        if not self.pool:
            self.pool = ConnectionPool.from_url(
                url=settings.redis_url,
                max_connections=settings.redis_n_connection_pool,
                socket_connect_timeout=5,
                socket_keepalive=True,
                health_check_interval=30,
            )
            self._semaphore = Semaphore(settings.redis_n_connection_pool)

    @asynccontextmanager
    async def get_redis(self) -> AsyncGenerator[Redis, None]:
        if not self.pool:
            await self.initialize()

        async with (
            self._semaphore
        ):  # add semaphore to wait for next available redis connection
            redis_client = Redis.from_pool(self.pool)
            try:
                yield redis_client
            except Exception as e:
                raise e
            finally:
                pass  # pass connection back to the pool

    async def close_redis(self):
        if self.pool:
            await self.pool.disconnect()

    async def is_ready(self):
        try:
            is_available = self.pool.can_get_connection()
            if is_available:
                print("Redis is connected")
            else:
                print("Redis is not connected")
        except ConnectionError:
            raise


REDIS_POOL = RedisConnectionPool()


async def get_redis() -> AsyncGenerator[Redis, None]:
    async with REDIS_POOL.get_redis() as redis:
        yield redis
