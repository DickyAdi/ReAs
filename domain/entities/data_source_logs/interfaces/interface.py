from abc import ABC, abstractmethod
from uuid import UUID

from ..data_source import DataSourceEntity


class DataSourceInterface(ABC):
    pass

    @abstractmethod
    async def create_data_source(self, data: DataSourceEntity): ...

    @abstractmethod
    async def remove_data_source(self, source_id: UUID): ...

    @abstractmethod
    async def get_data_source_by_id(self, source_id: UUID): ...
