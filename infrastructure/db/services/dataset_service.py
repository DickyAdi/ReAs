from sqlalchemy import select, func
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID

from ..error_mapper import DatabaseErrorMapper

from domain.entities.datasets import DatasetInterface, DatasetEntity, DatasetTypedDict
from domain.exceptions import (
    PaginationTypeError,
    DoubleIdentifierError,
)
from infrastructure.db.models.dataset_model import Datasets


class DatasetService(DatasetInterface):
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_dataset(self, dataset: DatasetEntity):
        obj = Datasets.from_entity(dataset)
        # async with self.uow as uow:
        try:
            self.db.add(obj)
        except SQLAlchemyError as e:
            raise DatabaseErrorMapper.map_error(e)
        return obj

    async def get_dataset_by_id(self, id: UUID):
        """Get dataset from database by id identifier

        Args:
            id (UUID): Id identifier for the searched dataset

        Raises:
            DatabaseErrorMapper: If SQLAlchemy raises error, this mapper will map to specific domain exception error

        Returns:
            Will return dataset ORM object or None if not found
        """
        stmt = select(Datasets).where(Datasets.id == id)
        # async with self.uow as uow:
        try:
            result = await self.db.execute(stmt)
            ds = result.scalar_one_or_none()
            return ds
        except SQLAlchemyError as e:
            raise DatabaseErrorMapper().map_error(e)

    async def get_dataset_by_name(self, name: str):
        """Get dataset from database by name identifier

        Args:
            name (str): Name of the dataset to be searched

        Raises:
            DatabaseErrorMapper: If SQLAlchemy raises error, this mapper will map to specific domain exception error

        Returns:
            Will return dataset ORM object if found or None if not found
        """
        stmt = select(Datasets).where(Datasets.name == name)
        # async with self.uow as uow:
        try:
            result = await self.db.execute(stmt)
            ds = result.scalar_one_or_none()
            return ds
        except SQLAlchemyError as e:
            raise DatabaseErrorMapper().map_error(e)

    async def get_datasets(
        self, limit: int = 10, offset: int = 0, id: UUID = None, name: str = None
    ):
        """Get datasets with pagination, if identifier is provided, will filter the corresponding dataset using the provided identifier

        Args:
            limit (int, optional): Number of data. Defaults to 10.
            offset (int, optional): Starting row index. Defaults to 0.
            id (UUID, optional): Search dataset using this id. Defaults to None.
            name (str, optional): Search dataset using this name. Defaults to None.

        Raises:
            PaginationTypeError: If `offset` is less than 0 and limit not between 1 - 100
            DatabaseErrorMapper.map_error: If SQLAlchemy raises error, this mapper will map to specific domain exception error

        Returns:
            Will return dict containing the total, offset, limit, and data
        """
        if id and name:
            DoubleIdentifierError(
                message="Must only choose 1 identifier", identifiers=["id", "name"]
            )

        if offset < 0:
            raise PaginationTypeError(
                message="`offset` must be non-negative integer", offset=offset
            )
        if limit < 1 and limit > 100:
            raise PaginationTypeError(
                message="`limit` must between 1 and 100", limit=limit
            )

        stmt = select(Datasets)
        if id:
            stmt = stmt.where(Datasets.id == id)

        if name:
            stmt = stmt.where(Datasets.name == name)

        if limit:
            stmt = stmt.limit(limit)
        if offset:
            stmt = stmt.offset(offset)

        count_stmt = select(func.count()).select_from(Datasets)
        # async with self.uow as uow:
        try:
            result = await self.db.execute(stmt)
            ds = result.scalars().all()
            if not offset and not limit:
                return ds
            count_value = await self.db.execute(count_stmt)
            count_res = count_value.scalar_one()
            return {
                "total": count_res,
                "offset": offset,
                "limit": limit,
                "data": ds,
            }
        except SQLAlchemyError as e:
            raise DatabaseErrorMapper.map_error(e)

    async def get_dataset_by_issuer(self, id: int, limit: int = 10, offset: int = 0):
        if offset < 0:
            raise PaginationTypeError(
                message="`offset` must be non-negative integer", offset=offset
            )
        if limit < 1 or limit > 100:
            raise PaginationTypeError(
                message="`limit` must between 1 and 100", limit=limit
            )

        stmt = select(Datasets).where(Datasets.issuer_id == id)
        count_stmt = (
            select(func.count()).select_from(Datasets).where(Datasets.issuer_id == id)
        )

        if offset:
            stmt = stmt.offset(offset)
        if limit:
            stmt = stmt.limit(limit)
        # async with self.uow as uow:
        try:
            res = await self.db.execute(stmt)
            ds = res.scalars().all()
            if not offset and not limit:
                return ds
            stats_res = await self.db.execute(count_stmt)
            stats = stats_res.scalar_one()
            return {"total": stats, "offset": offset, "limit": limit, "data": ds}
        except SQLAlchemyError as e:
            raise DatabaseErrorMapper.map_error(e)

    async def edit_dataset(self, dataset: Datasets, values: DatasetTypedDict):
        """Edit given dataset with the given dictionary, will edit the dataset instance from the given dictionary as is. Ensure dictionary keys match with the class property of `Datasets`

        Args:
            dataset (Datasets): Datasets instance that want to be edited
            values (DatasetTypedDict): New values

        Raises:
            DatabaseErrorMapper.map_error: Will raise specific domain error from the thrown SQLAlchemy error

        Returns:
            Edited dataset
        """
        [setattr(dataset, key, value) for key, value in values.items()]
        try:
            # async with self.uow as uow:
            self.db.add(dataset)
            # await uow.commit()
        except SQLAlchemyError as e:
            raise DatabaseErrorMapper.map_error(e)
        return dataset
