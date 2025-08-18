from dataclasses import dataclass, fields
from datetime import datetime
from typing import Optional

from domain.entities.user.interfaces import UserInterface
from domain.enums.users import Role, AuthProvider, UserStatus
from domain.enums.subscriptions import SubscriptionStatus, SubscriptionPlan
from domain.enums.tiers import Tier


# TODO: Refactor this. Move this to User entity or model
@dataclass
class PublicSubscriptionDTO:
    tier: Tier
    start_at: datetime
    end_at: datetime
    status: SubscriptionStatus
    billing_cycle: SubscriptionPlan


# TODO: Refactor this. Move this to User Entity or model.
@dataclass
class PublicUserDTO:
    name: str
    email: str
    is_validated: bool
    created_at: datetime
    updated_at: datetime
    role: Role
    status: UserStatus
    provider: AuthProvider
    subscription: PublicSubscriptionDTO
    # subscription:


class UserApplication:
    """User application for orchestrating user services."""

    def __init__(self, service: UserInterface):
        self.user_service = service

    async def edit_user_info(self, db, user, changed_data):
        """Edit users non-authentication/non-security related data.

        Args:
            db (Any): Infrastructure layer database session.
            user (Any): Infrastructure layer User ORM instance.
            changed_data (Any): Infrastructure Pydantic BaseModel.

        Returns:
            bool: Whether the user data is edited or not.
        """
        for key, value in changed_data.model_dump().items():
            setattr(user, key, value)
        updated_user = await self.user_service.edit_user(db=db, user=user)
        return updated_user

    # TODO: Tech debt. Refactor this later to use edit_user_info by adjusting the changed_email_value to pydantic base model.
    async def change_user_email(self, db, user, changed_email_value):
        """Change users email by modifying it inplace and pass the object to edit_user service.

        Args:
            db (Any): Infrastructure layer database session.
            user (Any): Infrastructure layer User ORM instance.
            changed_email_value (Any): Changed email value.

        Returns:
            User: Infrastructure layer User ORM instance.
        """
        user.email = changed_email_value
        if user.is_validated:
            user.is_validated = False
        updated_user = await self.user_service.edit_user(db=db, user=user)
        return updated_user

    async def change_user_password(self, db, user, hashed_changed_password_value):
        """Change user password by modifying it inplace and pass the object to edit_user service.

        Args:
            db (Any): Infrastrcuture layer database session.
            user (Any): Infrastructure layer User ORM instance.
            hashed_changed_password_value (Any): Hashed password value.

        Returns:
            User: Infrastructure layer User ORM instance.
        """
        user.password = hashed_changed_password_value
        user.token_version += 1
        updated_user = await self.user_service.edit_user(db=db, user=user)
        return updated_user

    async def get_user(
        self,
        db,
        email: Optional[str] = None,
        id: Optional[int] = None,
        with_sub: bool = False,
        role: Optional[Role] = Role.user,
    ):
        """Get user based on one of the identifier provided.

        Args:
            db (Any): Infrastructure layer database session.
            email (Optional[str], optional): Users email identifier. Defaults to None.
            id (Optional[int], optional): Users id identifier. Defaults to None.
            with_sub (bool, optional): Whether to enrich users data with subscriptions or not. Defaults to False.
            role (Optional[Role], optional): Filter by role. Defaults to Role.user.

        Raises:
            ValueError: If both `email` and `id` is passed. Only one identifier must be passed.

        Returns:
            User: Infrastructure layer User ORM instance.
        """
        if email and id:
            raise ValueError("Only 1 identifier must be provided.")
        if with_sub:
            user = await self.user_service.get_user_with_sub(db=db, id=id, email=email)
        else:
            user = await self.user_service.get_user_by_email(
                db=db, email=email, role=role
            )
        return user

    async def set_user_validated(self, db, user):
        """Set user email validated value.

        Args:
            db (Any): Infrastructure layer database session.
            user (User): Infrastructure layer User ORM instance.

        Returns:
            bool: Whether the user editting process is success or not.
        """
        user.is_validated = True
        is_edited = await self.user_service.edit_user(db=db, user=user)
        return is_edited

    # TODO: Refactor casting value to public DTO by calling a method in the user entity or model
    async def get_users(
        self,
        db,
        offset: int,
        limit: int,
        role: Optional[Role] = Role.user,
        with_sub: bool = False,
    ):
        """Return list of users publicly save data with DTO.

        Args:
            db (Any): Infrastructure layer database session.
            offset (int): Starts row....
            limit (int): Limit returned/shown row....
            role (Optional[Role], optional): Filter by role. Defaults to Role.user.
            with_sub (bool, optional): Whether to enrich users data with subscriptions value or not. Defaults to False.

        Raises:
            TypeError: If `offset` or `limit` is not type of Integer.

        Returns:
            List[User]: List of Infrastructure layer Users ORM instances.
        """
        if not isinstance(offset, int) or not isinstance(limit, int):
            raise TypeError(
                f"Offset and or limit must be type Integer. Got {type(offset)} and {type(limit)}"
            )
        users = await self.user_service.get_users(
            db=db, offset=offset, limit=limit, role=role, with_sub=with_sub
        )
        user_dto = {f.name for f in fields(PublicUserDTO)}
        for i in range(len(users["data"])):
            user = users["data"][i]
            user_subs = user.subscriptions[0]
            sub = PublicSubscriptionDTO(
                start_at=user_subs.start_at,
                end_at=user_subs.end_at,
                status=user_subs.status,
                tier=user_subs.plan.tier.name,
                billing_cycle=user_subs.plan.billing_cycle,
            )
            usr = PublicUserDTO(
                **{
                    key: (getattr(user, key) if key != "subscription" else sub)
                    for key in user_dto
                }
            )  # * Manually assign subscription attr
            usr.subscription = sub
            users["data"][i] = usr
        return users
