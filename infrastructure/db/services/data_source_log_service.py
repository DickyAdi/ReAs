from sqlalchemy.ext.asyncio import AsyncSession

from domain.entities.data_source_logs import DataSourceInterface


class DataSourceService(DataSourceInterface):
    def __init__(self, db: AsyncSession):
        pass
