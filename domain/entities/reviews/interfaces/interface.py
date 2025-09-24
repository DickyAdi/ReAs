from abc import ABC, abstractmethod
from typing import Union, Optional
from uuid import UUID

from domain.entities.reviews import ReviewEntity


class ReviewInterface(ABC):
    @abstractmethod
    async def create_review(
        self,
        review: Union[list[ReviewEntity], ReviewEntity],
        commit: Optional[bool] = True,
        generate_defaults: Optional[bool] = False,
    ): ...

    @abstractmethod
    async def batch_insert(self, reviews): ...
    @abstractmethod
    async def get_reviews_by_dataset_id(
        self,
        dataset_id: UUID,
        offset: Optional[int] = 0,
        limit: Optional[int] = 0,
    ): ...
