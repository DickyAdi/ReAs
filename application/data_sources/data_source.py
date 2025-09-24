from uuid import UUID

from domain.enums.datasets import DatasetProvider
from domain.entities.data_source_logs import DataSourceEntity
from domain.uow import UnitOfWorkInterface


class DataSourceApplication:
    def __init__(self, uow: UnitOfWorkInterface):
        self.uow = uow

    async def log_data_source(
        self, provider: DatasetProvider, additional_metadata: dict = None
    ):
        """Create data source log in the database. This method must be called within UoW context as this method returns added ORM object to the database session. See usages

        Args:
            provider (DatasetProvider): Used dataset provider enum
            additional_metadata (dict, optional): Additional metadata if needed to store it within the database. Defaults to None.

        Usages:
            ```python
            > #caller
            > data_source_app = DataSourceApplication(...)
            > async with uow:
            >   ...
            >   data_source = await data_source_app.log_data_source(DatasetProvider.scrape)
            >   ...
            ```

        Raises:
            RunTimeError: If method is not called inside UnitOfWork context.

        Returns:
            Added data source log ORM object to the session
        """
        if not self.uow.in_transaction:
            raise RuntimeError(
                "`log_data_source` method must be called inside UnitOfWork context"
            )
        domain_data = DataSourceEntity.create(
            provider=provider, additional_metadata=additional_metadata or None
        )
        try:
            data_source = await self.uow.data_sources.create_data_source(
                data=domain_data
            )
            await self.uow.add(
                data_source  # * only add, let caller handle the commit or anything else.
            )
            return data_source
        except Exception:
            raise

    async def remove_data_source(self, source_id: UUID):
        try:
            is_removed = await self.uow.data_sources.remove_data_source(
                source_id=source_id
            )
            return is_removed
        except Exception:
            raise
