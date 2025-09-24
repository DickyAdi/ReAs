from uuid import UUID
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from domain.entities.data_source_logs import DataSourceInterface, DataSourceEntity
from domain.exceptions import DataSourceNotFoundError
from infrastructure.db.models.data_source_log_model import DataSource


class DataSourceService(DataSourceInterface):
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_data_source(self, data: DataSourceEntity):
        """Create ORM data source object

        Args:
            data (DataSourceEntity): Domain data source object

        Returns:
            Created ORM data source object from domain data source object
        """
        obj = DataSource.from_entity(data_source=data)
        return obj

    async def get_data_source_by_id(self, source_id: UUID):
        stmt = select(DataSource).where(DataSource.id == source_id)
        result = await self.db.execute(stmt)
        data_source = result.scalar_one_or_none()
        return data_source

    async def remove_data_source(self, source_id: UUID) -> bool:
        data_source = await self.get_data_source_by_id(source_id=source_id)
        if not data_source:
            raise DataSourceNotFoundError(identifier=source_id)
        stmt = delete(DataSource).where(DataSource.id == source_id)
        await self.db.execute(
            stmt
        )  # * only execute, leave commit to Unit of Work (UoW)
        return True
