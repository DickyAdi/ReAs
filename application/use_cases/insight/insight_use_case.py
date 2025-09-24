from uuid import UUID
from typing import Optional

from application.insights import InsightApplication
from application.datasets import DatasetApplication

from domain.uow import UnitOfWorkInterface
from domain.exceptions import DatasetNotFoundError


class InsightsUseCase:
    def __init__(self, uow: UnitOfWorkInterface):
        self.uow = uow

    async def get_review_insight(
        self, dataset_id: UUID, offset: Optional[int] = 0, limit: Optional[int] = 0
    ):
        async with self.uow as u:
            insight_app = InsightApplication(uow=u)
            dataset_app = DatasetApplication(uow=u)
            dataset = await dataset_app.get_dataset(id=dataset_id)
            if not dataset:
                raise DatasetNotFoundError(identifier=dataset_id)
            if dataset.is_empty:
                return []
            data = await insight_app.get_review_insight(
                dataset_id=dataset_id, offset=offset, limit=limit
            )
            return data
