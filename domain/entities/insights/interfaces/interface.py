from abc import ABC, abstractmethod
from typing import Optional, Any
from uuid import UUID

from domain.entities.insights import InsightEntity


class InsightInterface(ABC):
    @abstractmethod
    async def create_insight(
        self,
        obj: InsightEntity,
        commit: Optional[bool] = True,
        generate_defaults: Optional[bool] = False,
    ): ...

    @abstractmethod
    async def upsert_insights(self, insights: list[Any]): ...

    @abstractmethod
    async def upsert_assoc_table(self, assoc_values: list[dict]): ...

    @abstractmethod
    async def get_insights_by_dataset_id(
        self, dataset_id: UUID, offset: Optional[int] = 0, limit: Optional[int] = 0
    ): ...

    @abstractmethod
    async def get_review_insight(
        self, dataset_id: UUID, offset: Optional[int] = 0, limit: Optional[int] = 0
    ): ...
