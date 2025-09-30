from abc import ABC, abstractmethod
from typing import Optional


class ScrapingInterface(ABC):
    resource: str = None

    @abstractmethod
    def build(self, base_url: str, resource_url: str): ...

    @abstractmethod
    def hit(self, headers: Optional[dict] = None): ...

    def build_hit(self, base_url, resource_url: str, headers: Optional[dict]):
        self.build(
            base_url=base_url,
            resource_url=resource_url,
        )
        return self.hit(headers=headers)
