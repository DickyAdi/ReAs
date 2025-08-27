from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from sqlalchemy.exc import SQLAlchemyError
from typing import Optional, List, Dict, Any

from domain.exceptions import DoubleIdentifierError, PaginationTypeError
from domain.entities.tiers.interfaces import TiersInterface
from domain.entities.tiers import TierEntity
from domain.enums.tiers import Tier
from ...db.models import Tiers
from ...db.error_mapper import DatabaseErrorMapper


class TierService(TiersInterface):
    async def get_tier_by_name(self, db: AsyncSession, tier_name: Tier) -> Tiers | None:
        """Get tier using tier name identifier.

        Args:
            db (AsyncSession): Async session of the database.
            tier_name (str): tier name via tier enum.

        Returns:
            Union[Tiers, None]: If Tiers with the name exists, will return corresponding Tiers. Else, return None.
        """
        statement = select(Tiers).where(Tiers.name == tier_name)
        try:
            result = await db.execute(statement)
        except SQLAlchemyError as e:
            raise DatabaseErrorMapper().map_error(e)
        tier = result.scalars().first()
        return tier

    async def get_tier_by_id(self, db: AsyncSession, id: int) -> Tiers | None:
        """Get tier using id identifier.

        Args:
            db (AsyncSession): Async session of the database.
            id (int): tier identifier.

        Returns:
            Union[Tiers, None]: If Tiers with the id exists, will return corresponding Tiers. Else, return None.
        """
        statement = select(Tiers).where(Tiers.id == id)
        try:
            result = await db.execute(statement)
        except SQLAlchemyError as e:
            raise DatabaseErrorMapper().map_error(e)
        tier = result.scalars().first()
        return tier

    async def get_tiers(
        self, db: AsyncSession, offset: int = 0, limit: int = 0
    ) -> List[Tiers] | Dict[str, Any]:
        """Get all available tiers with pagination wise via `offset` and `limit`.
        If `offset` and `limit` are both 0, it will return all available tiers.

        Args:
            db (AsyncSession): Async session of the database.
            offset (int, optional): Start row of select statement. Defaults to 0.
            limit (int, optional): Limit results of select statement. Defaults to 0.

        Returns:
            Union[List[Tiers], Dict[str, Any]]:
                - Return list of Tiers if `offset` or `limit` are 0.
                - Return dict with total count, offset, limit and paginated data otherwise.
        """
        if offset < 0:
            raise PaginationTypeError(
                message="`offset` must be non-negative integer value", offset=offset
            )
        if limit < 1 and limit > 100:
            raise PaginationTypeError(
                message="`limit` must not be less than 1 and less than 100", limit=limit
            )
        statement = select(Tiers)
        if offset:
            statement = statement.offset(offset)
        if limit:
            statement = statement.limit(limit)
        try:
            result = await db.execute(statement)
        except SQLAlchemyError as e:
            raise DatabaseErrorMapper.map_error(e)
        tiers = result.scalars().all()

        if not offset and not limit:
            return tiers

        count_stmt = select(func.count()).select_from(Tiers)
        try:
            count_result = await db.execute(count_stmt)
        except SQLAlchemyError as e:
            raise DatabaseErrorMapper().map_error(e)
        count_value = count_result.scalar_one()
        return {"total": count_value, "offset": offset, "limit": limit, "data": tiers}

    async def create_tier(
        self, db: AsyncSession, tier: TierEntity, commit=True
    ) -> Tiers:
        """Create Tier based on domain entity.

        Args:
            db (AsyncSession): Async session of the database.
            tier (TierEntity): New tier to be created/inserted to db.
            commit (bool): Should commit flag.

        Returns:
            Tiers: New Tiers ORM obj.
        """
        new_tier = Tiers.from_entity(tier)
        try:
            db.add(new_tier)
            await db.flush()
            if commit:
                await db.commit()
                await db.refresh(new_tier)
            return new_tier
        except SQLAlchemyError as e:
            raise DatabaseErrorMapper().map_error(e)

    async def delete_tier(
        self,
        db: AsyncSession,
        tier_name: Optional[str] = None,
        tier_id: Optional[int] = None,
    ) -> bool:
        """Delete tier based on either tier name or tier id. Only 1 identifier must be provided, or else it will raise KeyError.

        Args:
            db (AsyncSession): Async session of the database.
            tier_name (Optional[str], optional): Tier name identifier that want to be deleted. Defaults to None.
            tier_id (Optional[int], optional): Tier id identifier that want to be deleted. Defaults to None.

        Raises:
            KeyError: Cannot use 2 identifiers at the same time, choose only 1.

        Returns:
            True | False: Return deletion state. True means tier was deleted, False otherwise.
        """
        if tier_name and tier_id:
            # raise KeyError("Cannot delete tier using 2 identifiers provided. Only 1 identifier must be provided.")
            raise DoubleIdentifierError(identifiers=["tier_name", "tier_id"])
        if tier_name:
            stmt = select(Tiers).where(Tiers.name == tier_name)
        elif tier_id:
            stmt = select(Tiers).where(Tiers.id == tier_id)
        try:
            res = await db.execute(stmt)
            tier_db = res.scalar_one_or_none()
            if tier_db:
                await db.delete(tier_db)
                await db.commit()
                return True
            return False
        except SQLAlchemyError as e:
            raise DatabaseErrorMapper().map_error(e)

    async def edit_tier(self, db: AsyncSession, tier: Tiers) -> bool:
        """Editting tier based on Tiers ORM Obj.
        Edit desired edited tier and then pass it to this method.

        Args:
            db (AsyncSession): Async session of the database.
            tier (Tiers): Modified Tiers obj

        Returns:
            True | False: Return the bool of editting state. True means edit is success, False otherwise.
        """
        try:
            if tier:
                await db.commit()
                await db.refresh(tier)
                return True
            return False
        except SQLAlchemyError as e:
            raise DatabaseErrorMapper().map_error(e)
