from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from infrastructure.db import UnitOfWork, get_db
from infrastructure.db.services.dataset_service import DatasetService
from application.datasets import DatasetApplication


def get_dataset_app(db: AsyncSession = Depends(get_db)) -> "DatasetApplication":
    return DatasetApplication(uow=UnitOfWork(db=db))
    # return DatasetApplication(service=DatasetService(uow=UnitOfWork(db=db)))
