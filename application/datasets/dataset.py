from typing import Optional, Any
from uuid import UUID

from domain.entities.datasets import DatasetEntity
from domain.exceptions import (
    DatasetNotFoundError,
    DoubleIdentifierError,
    UniqueConstraintViolationError,
    DatasetNameAlreadyExistError,
)
from domain.uow import UnitOfWorkInterface
from application.rules import enforce_transaction


class DatasetApplication:
    def __init__(
        self,
        uow: UnitOfWorkInterface,
    ):
        self.uow = uow

    @enforce_transaction
    async def create_dataset(
        self, dataset_name: str, issuer_id: int, commit: Optional[bool] = True
    ):
        domain_dataset = DatasetEntity.create(name=dataset_name, issuer_id=issuer_id)
        async with self.uow as uow:
            try:
                dataset = await uow.datasets.create_dataset(domain_dataset)
                if commit:
                    await uow.commit()
                return dataset
            except UniqueConstraintViolationError:
                raise DatasetNameAlreadyExistError(name=dataset_name)

    @enforce_transaction
    async def edit_dataset(
        self,
        changed_value: dict,
        id: Optional[UUID] = None,
        name: Optional[str] = None,
        commit: Optional[bool] = True,
    ):
        if id and name:
            raise DoubleIdentifierError(
                "Must only chose 1 identifier", identifiers=["id", "name"]
            )
        ds = None
        if id:
            ds = await self.uow.datasets.get_dataset_by_id(id)
        else:
            ds = await self.uow.datasets.get_dataset_by_name(name)
        if not ds:
            raise DatasetNotFoundError(identifier=id or name)
        edited_dataset = await self.uow.datasets.edit_dataset(
            dataset=ds, values=changed_value
        )
        if commit:
            await self.uow.commit()
        return edited_dataset

    @enforce_transaction
    async def get_dataset(
        self, id: Optional[UUID] = None, name: Optional[str] = None
    ) -> Any:
        if id and name:
            raise DoubleIdentifierError(
                "Must only chose 1 identifier", identifiers=["id", "name"]
            )
        ds = None
        if id:
            ds = await self.uow.datasets.get_dataset_by_id(id)
        else:
            ds = await self.uow.datasets.get_dataset_by_name(name)
        if not ds:
            raise DatasetNotFoundError(identifier=id or name)
        return ds
