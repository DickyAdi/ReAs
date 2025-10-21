from concurrent.futures import ThreadPoolExecutor

from infrastructure.registry.model import ModelRegistry
from infrastructure.redis import REDIS_POOL
from config.settings import settings


def load_model(model_name: str = "default"):
    registry = ModelRegistry()
    return registry.get_model(model_name)


def get_executor(max_workers=settings.concurrent_worker):
    return ThreadPoolExecutor(max_workers=max_workers)


async def start_app():
    await REDIS_POOL.initialize()
    await REDIS_POOL.is_ready()


async def stop_app():
    await REDIS_POOL.close_redis()
