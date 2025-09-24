from abc import ABC, abstractmethod
from uuid import UUID
from typing import Optional

from ..dataset import DatasetEntity, DatasetTypedDict


class DatasetInterface(ABC):
    @abstractmethod
    async def create_dataset(self, dataset: DatasetEntity): ...

    @abstractmethod
    async def get_dataset_by_name(self, name: str): ...

    @abstractmethod
    async def get_dataset_by_id(self, id: UUID): ...

    @abstractmethod
    async def get_datasets(self, limit: int, offset: int): ...

    @abstractmethod
    async def get_dataset_by_issuer(self, id: int, email: str): ...

    @abstractmethod
    async def edit_dataset(self, dataset, values: DatasetTypedDict): ...
