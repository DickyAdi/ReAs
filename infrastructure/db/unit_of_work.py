from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError
from typing import Any, Union

from .services import DatasetService, ReviewService, DataSourceService, InsightService
from .error_mapper import DatabaseErrorMapper


class UnitOfWork:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def __aenter__(self):
        self.data_sources = DataSourceService(db=self.db)
        self.datasets = DatasetService(db=self.db)
        self.insights = InsightService(db=self.db)
        self.reviews = ReviewService(db=self.db)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_trace):
        if exc_type:
            await self.rollback()

    async def commit(self):
        try:
            await self.db.commit()
        except SQLAlchemyError as e:
            raise DatabaseErrorMapper().map_error(e)

    async def rollback(self):
        try:
            await self.db.rollback()
        except SQLAlchemyError as e:
            raise DatabaseErrorMapper().map_error(e)

    async def flush(self):
        try:
            await self.db.flush()
        except SQLAlchemyError as e:
            raise DatabaseErrorMapper().map_error(e)

    async def add(self, obj: Union[Any, list[Any]]):
        try:
            if isinstance(obj, list):
                self.db.add_all(obj)
            else:
                self.db.add(obj)
        except SQLAlchemyError as e:
            raise DatabaseErrorMapper().map_error(e)
