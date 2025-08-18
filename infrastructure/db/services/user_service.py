from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from sqlalchemy.orm import contains_eager
from typing import Optional
from uuid import UUID

from domain.entities.user import UserEntities
from domain.entities.user.interfaces import UserInterface
from ...db.models import User
from ...db.models import Subscriptions
from ...db.models import SubscriptionPlans
from domain.enums.users import Role
from domain.enums.subscriptions import SubscriptionStatus


class UserService(UserInterface):
    """Infrastructure layer user services.

    Args:
        UserInterface (Class): Domain user services contract.
    """

    async def get_users(
        self,
        db: AsyncSession,
        offset: int = 0,
        limit: int = 0,
        role: Optional[Role] = Role.user,
        with_sub: bool = False,
    ):
        """Retrieve users with pagination.

        Args:
            db (AsyncSession): Database session.
            offset (int, optional): Starts row..... Defaults to 0.
            limit (int, optional): Limit shown/retrieved row or data.... Defaults to 0.
            role (Optional[Role], optional): Filter by role. Defaults to Role.user.
            with_sub (bool, optional): Whether to enrich retrieved user data with their subscriptions or not. Defaults to False.

        Returns:
            List[Users]: List of Users data with number of limit.
        """
        statement = select(User)
        if with_sub:
            statement = (
                statement.join(Subscriptions)
                .filter(Subscriptions.status == SubscriptionStatus.active)
                .options(
                    contains_eager(User.subscriptions)
                    .joinedload(Subscriptions.plan)
                    .joinedload(SubscriptionPlans.tier)
                )
            )
        if offset:
            statement = statement.offset(offset)
        if limit:
            statement = statement.limit(limit)
        if role:
            statement = statement.where(User.role == role)
        result = await db.execute(statement)
        if with_sub:
            users = result.unique().scalars().all()
        else:
            users = result.scalars().all()

        if not offset and not limit:
            return users

        count_statement = (
            select(func.count()).select_from(User).where(User.role == role)
        )
        count_result = await db.execute(count_statement)
        count_value = count_result.scalar_one()
        return {"total": count_value, "offset": offset, "limit": limit, "data": users}

    async def get_user_by_email(
        self, db: AsyncSession, email: str, role: Optional[Role] = Role.user
    ):
        """Retrieve/get user data by email.

        Args:
            db (AsyncSession): Database session.
            email (str): email identifier.
            role (Optional[Role], optional): Filter by role. Defaults to Role.user.

        Returns:
            User: Retrieved User ORM instance.
        """
        statement = select(User).where(User.email == email)
        if role:
            statement = statement.where(User.role == role)
        result = await db.execute(statement)
        user = result.scalars().first()
        return user

    async def get_user_by_uuid(
        self, db: AsyncSession, uuid: UUID, role: Optional[Role] = Role.user
    ):
        """Retrieve/get user data by users UUID.

        Args:
            db (AsyncSession): Database session.
            uuid (UUID): Users UUID identifier.
            role (Optional[Role], optional): Filter by row. Defaults to Role.user.

        Returns:
            User: User ORM instance.
        """
        stmt = select(User).where(User.uuid == uuid)
        if role:
            stmt = stmt.where(User.role == role)
        res = await db.execute(statement=stmt)
        user = res.scalars().first()
        return user

    async def create_user(
        self, db: AsyncSession, user: UserEntities, commit: bool = True
    ):
        """Create new user based on User domain entity.

        Args:
            db (AsyncSession): Database session.
            user (UserEntities): User domain entity.
            commit (bool, optional): Whether to commit DML to the database or not. Defaults to True.

        Returns:
            User: Created new user ORM instance.
        """
        new_user = User.from_entity(user)
        db.add(new_user)
        await db.flush()
        if commit:
            await db.commit()
            await db.refresh(new_user)
        return new_user

    async def delete_user(
        self,
        db: AsyncSession,
        email: Optional[str] = None,
        user_id: Optional[int] = None,
    ):
        """Delete user by one of the given identifiers.

        Args:
            db (AsyncSession): Database session.
            email (Optional[str], optional): Users email identifier. Defaults to None.
            user_id (Optional[int], optional): Users id identifier. Defaults to None.

        Raises:
            KeyError: If `email` and `user_id` was provided. Only one must be provided.

        Returns:
            bool: Whether user has been deleted or not.
        """
        if email and user_id:
            raise KeyError(
                "Cannot delete user using 2 identifiers provided. Only 1 identifier must be provided."
            )
        if email:
            statement = select(User).where(User.email == email)
        elif user_id:
            statement = select(User).where(User.id == user_id)
        result = await db.execute(statement)
        user_db = result.scalar_one_or_none()
        if user_db:
            await db.delete(user_db)
            await db.commit()
            return True
        return False

    async def edit_user(self, db: AsyncSession, user: User):
        """Editting user based on User ORM Obj.

        Args:
            db (AsyncSession): Async session of the database.
            user (User): Modified User obj.

        Returns:
            True | False: Return the bool of editting, True means edit value has been applied, False otherwise.
        """
        if user:
            await db.commit()
            await db.refresh(user)
            return True
        return False

    async def get_user_with_sub(
        self, db: AsyncSession, id: Optional[int], email: Optional[str]
    ):
        if id and email:
            raise ValueError("Only 1 identifier must be provided.")
        # * Statement below assuming each user only has 1 active subscriptions.
        stmt = (
            select(User)
            .join(Subscriptions)
            .filter(Subscriptions.status == SubscriptionStatus.active)
            .options(
                contains_eager(User.subscriptions)
                .joinedload(Subscriptions.plan)
                .joinedload(SubscriptionPlans.tier)
            )
        )
        if id:
            stmt = stmt.where(User.id == id)
        else:
            stmt = stmt.where(User.email == email)
        res = await db.execute(stmt)
        return res.scalars().first()
