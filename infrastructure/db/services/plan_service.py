from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import contains_eager
from sqlalchemy import select, func, desc
from sqlalchemy.exc import SQLAlchemyError
from typing import Dict, List, Any

from ..models.subscriptions_model import SubscriptionPlans, Tiers
from ...db.error_mapper import DatabaseErrorMapper

from domain.entities.subscriptions.interfaces import SubscriptionPlansInterface
from domain.entities.subscriptions import SubscriptionPlansEntity
from domain.exceptions import PaginationTypeError


class SubscriptionPlanService(SubscriptionPlansInterface):
    async def create_plans(
        self, db: AsyncSession, plan: SubscriptionPlansEntity, commit: bool = True
    ) -> SubscriptionPlans:
        """Create a new plan based on domain entity.

        Args:
            db (AsyncSession): Async session of the database
            plan (SubscriptionPlansEntity): Domain entity

        Returns:
            SubscriptionPlans: ORM Obj of the newly created plan.
        """
        new_plan = SubscriptionPlans.from_entity(plan)
        try:
            db.add(new_plan)
            await db.flush()
            if commit:
                await db.commit()
                await db.refresh(new_plan)
            return new_plan
        except SQLAlchemyError as e:
            raise DatabaseErrorMapper().map_error(e)

    async def edit_plans(self, db: AsyncSession, plan: SubscriptionPlans) -> bool:
        """Edit plans based on edited ORM Obj.

        Args:
            db (AsyncSession): Async session of the database.
            plan (SubscriptionPlans): Edited ORM Obj

        Usage:
            ```python
            ToBeEdited.editedKeys = updatedValue
            print(InitObj.edit_plans(db=TheDB, plan=ToBeEdited))
            True
            ```

        Returns:
            True | False: Return the bool of editting, True means edit value has been applied, False otherwise.
        """
        try:
            if plan:
                await db.commit()
                await db.refresh(plan)
                return True
            return False
        except SQLAlchemyError as e:
            raise DatabaseErrorMapper().map_error(e)

    async def get_plans(
        self,
        db: AsyncSession,
        offset: int = 0,
        limit: int = 0,
        with_tier: bool = False,
        isActive: bool = False,
    ) -> List[SubscriptionPlans] | Dict[str, Any]:
        """Get plans with pagination, if `offset` or `limit` provided pagination will be applied.
        Else, return all plans available.

        Args:
            db (AsyncSession): Async session of the database.
            offset (int, optional): Desired starts row. Defaults to 0.
            limit (int, optional): Desired limit from the select statement. Defaults to 0.
            with_tier (bool): Whether to join plan with tier or not. Defaults to False.
            isActive (bool): Whether to only show active plan or not. Defaults to False.
        Returns:
            List[SubscriptionPlans] | Dict[str, Any]: Will return list of the ORM Obj if pagination variable are set to 0,
            else will return dict that includes total count, used offset, used limit, and data from the query.
        """
        if offset < 0:
            raise PaginationTypeError(
                message="`offset` must be non-negative integer value", offset=offset
            )
        if limit < 1 and limit > 100:
            raise PaginationTypeError(
                message="`limit` must not be less than 1 and less than 100", limit=limit
            )
        stmnt = select(SubscriptionPlans)
        if with_tier:
            stmnt = stmnt.join(Tiers).options(contains_eager(SubscriptionPlans.tier))
        if isActive:
            stmnt = stmnt.filter(SubscriptionPlans.is_active == True)  # noqa
        if offset:
            stmnt = stmnt.offset(offset)
        if limit:
            stmnt = stmnt.limit(limit)
        try:
            res = await db.execute(stmnt)
        except SQLAlchemyError as e:
            raise DatabaseErrorMapper().map_error(e)
        if with_tier:
            plans = res.unique().scalars().all()
        else:
            plans = res.scalars().all()

        if not offset and not limit:
            return plans

        if isActive:
            count_stmnt = (
                select(func.count())
                .select_from(SubscriptionPlans)
                .where(SubscriptionPlans.is_active == isActive)
            )
        else:
            count_stmnt = select(func.count()).select_from(SubscriptionPlans)
        try:
            count_result = await db.execute(count_stmnt)
        except SQLAlchemyError as e:
            raise DatabaseErrorMapper().map_error(e)
        count_value = count_result.scalar_one()
        return {"total": count_value, "offset": offset, "limit": limit, "data": plans}

    async def get_plan_by_code(
        self, db: AsyncSession, code: str, is_latestActive: bool = True
    ) -> SubscriptionPlans | None | List[SubscriptionPlans]:
        """Get plan by unique code, code must be a string in this uppercase `Tier.name-billing_cycle-currency` format.

        Args:
            is_latestActive (bool): Return the latest and active plan. Defaults to True.
            db (AsyncSession): Async session of the database.
            code (str): Unique formatted code for the plan.

        Returns:
            SubscriptionPlans | None | List[SubscriptionPlans]:
            - Return single instance of ORM Obj if is_latestActive is True.
            - Return List of SubscriptionPlans if is_latestActive is False.
            - Return None if no code is match.
        """
        stmnt = select(SubscriptionPlans).where(SubscriptionPlans.code == code)
        if is_latestActive:
            stmnt = (
                stmnt.order_by(desc(SubscriptionPlans.created_at))
                .where(SubscriptionPlans.is_active == True)  # noqa
                .limit(1)
            )
        try:
            res = await db.execute(stmnt)
        except SQLAlchemyError as e:
            raise DatabaseErrorMapper().map_error(e)
        plans = res.scalars()
        if is_latestActive:
            return plans.first()
        return plans.all()

    async def deactivate_plan(self, db: AsyncSession, plan: SubscriptionPlans) -> bool:
        try:
            plan.is_active = False
            await db.flush()
            return True
        except SQLAlchemyError as e:
            raise DatabaseErrorMapper().map_error(e)
