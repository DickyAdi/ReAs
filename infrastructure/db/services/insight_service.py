from sqlalchemy.ext.asyncio import AsyncSession

from domain.entities.insights import InsightInterface


class InsightService(InsightInterface):
    def __init__(self, db: AsyncSession):
        pass
