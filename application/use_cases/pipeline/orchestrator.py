from typing import Optional, Tuple, Any
from uuid import UUID

from application.datasets import DatasetApplication
from application.data_sources import DataSourceApplication
from domain.workers import WorkerInterface
from domain.uow import UnitOfWorkInterface
from domain.enums.datasets import DatasetStatus, DatasetProvider
from domain.enums.texts import TextLanguage


class DatasetPipelineOrchestrator:
    def __init__(self, service: WorkerInterface, uow: UnitOfWorkInterface):
        self.service = service
        self.uow = uow

    async def log_ingress_data(
        self,
        provider: DatasetProvider,
        dataset_id: Optional[UUID] = None,
        dataset_name: Optional[str] = None,
    ) -> Tuple[Any, Any]:
        async with self.uow as uow:
            dataset_app = DatasetApplication(uow)
            datasource_app = DataSourceApplication(uow)
            status = {"status": DatasetStatus.processing}
            edited_dataset = await dataset_app.edit_dataset(
                changed_value=status, id=dataset_id, name=dataset_name, commit=False
            )
            source_log = await datasource_app.log_data_source(provider=provider)
            await uow.commit()
        return edited_dataset, source_log

    async def rollback_ingress_data(self, dataset_id: UUID, source_id: UUID):
        async with self.uow as uow:
            dataset_app = DatasetApplication(uow)
            datasource_app = DataSourceApplication(uow)
            status = {"status": DatasetStatus.failed}
            edited_dataset = await dataset_app.edit_dataset(
                changed_value=status, id=dataset_id, commit=False
            )
            source_log_is_removed = await datasource_app.remove_data_source(
                source_id=source_id
            )
            await uow.commit()
            if edited_dataset and source_log_is_removed:
                return True
        return False

    async def run(
        self,
        source: DatasetProvider,
        language: TextLanguage,
        dataset_id: Optional[UUID] = None,
        dataset_name: Optional[str] = None,
        **kwargs,
    ):
        # * pass model name to the celery worker, which in this case is self.service.run()
        edited_dataset, source_log = await self.log_ingress_data(
            provider=source,
            dataset_id=dataset_id,
            dataset_name=dataset_name,
        )
        task_id = self.service.run(
            dataset_id=edited_dataset.id,
            provider_ref=source_log.id,
            language=language,
            model_name="default",
            **kwargs,
        )
        return task_id
