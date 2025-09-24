from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError
from typing import Any, Union

from .services import DatasetService, ReviewService, DataSourceService, InsightService
from .error_mapper import DatabaseErrorMapper


class UnitOfWork:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.in_transaction = False

    async def __aenter__(self):
        self.data_sources = DataSourceService(db=self.db)
        self.datasets = DatasetService(db=self.db)
        self.insights = InsightService(db=self.db)
        self.reviews = ReviewService(db=self.db)
        self.in_transaction = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_trace):
        self.in_transaction = False
        if exc_type:
            await self.rollback()
        await self.db.close()

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

    async def execute(self, statement):
        try:
            await self.db.execute(statement)
        except SQLAlchemyError as e:
            raise DatabaseErrorMapper.map_error(e)

    async def add(self, obj: Union[Any, list[Any]]):
        try:
            if isinstance(obj, list):
                self.db.add_all(obj)
            else:
                self.db.add(obj)
        except SQLAlchemyError as e:
            raise DatabaseErrorMapper().map_error(e)

    async def close(self):
        try:
            await self.db.close()
        except SQLAlchemyError as e:
            raise DatabaseErrorMapper.map_error(e)
