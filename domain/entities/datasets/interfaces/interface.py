from abc import ABC, abstractmethod
from uuid import UUID
from ..dataset import DatasetEntity


class DatasetInterface(ABC):
    @abstractmethod
    async def create_dataset(self, dataset: DatasetEntity): ...

    @abstractmethod
    async def get_dataset_by_name(self, name: str): ...

    @abstractmethod
    async def get_dataset_by_id(self, id: UUID): ...

    @abstractmethod
    async def get_datasets(self, limit: int, offset: int): ...
