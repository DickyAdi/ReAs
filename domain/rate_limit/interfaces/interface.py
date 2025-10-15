from abc import ABC, abstractmethod


class RateLimitInterface(ABC):
    @abstractmethod
    async def is_allowed(self, key: str):
        pass
