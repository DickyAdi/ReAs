from domain.entities.datasets import DatasetEntity
from domain.exceptions import (
    DatasetNameAlreadyExistError,
    UniqueConstraintViolationError,
)
from domain.uow import UnitOfWorkInterface


class DatasetApplication:
    def __init__(
        self,
        # service:DatasetInterface,
        uow: UnitOfWorkInterface,
    ):
        # self.service = service
        self.uow = uow

    async def create_dataset(self, dataset_name: str, issuer_id: int):
        domain_dataset = DatasetEntity.create(name=dataset_name, issuer_id=issuer_id)
        async with self.uow as uow:
            # dataset = await uow.datasets.create_dataset(domain_dataset)
            # print(f"[DEBUG/application/dataset.py] Printing domain dataset {domain_dataset.id}")
            try:
                dataset = await uow.datasets.create_dataset(domain_dataset)
                await uow.add(dataset)
                await uow.commit()
                return dataset
            except UniqueConstraintViolationError:
                raise DatasetNameAlreadyExistError(name=dataset_name)
