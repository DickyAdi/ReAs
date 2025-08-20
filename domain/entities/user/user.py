from uuid import UUID
from dataclasses import dataclass
from typing import Optional, List, TYPE_CHECKING
from datetime import datetime

if TYPE_CHECKING:
    from domain.entities.subscriptions import SubscriptionEntity
    from domain.entities.datasets import DatasetEntity

from domain.enums.users import AuthProvider, Role, UserStatus


@dataclass
class UserEntities:
    name: str
    email: str
    provider: AuthProvider
    status: UserStatus
    token_version: int
    password: Optional[str] = None
    role: Optional[Role] = Role.user
    id: Optional[int] = None  # derived from database
    uuid: Optional[UUID] = None  # derived from database
    current_tier_id: Optional[int] = (
        None  # derived from database FK on subscriptions.tier_id
    )
    created_at: Optional[datetime] = None  # derived from database
    updated_at: Optional[datetime] = None  # derived from database
    is_validated: Optional[bool] = False

    # related entities
    subscriptions: Optional[List["SubscriptionEntity"]] = None
    dataset: Optional[List["DatasetEntity"]] = None

    @classmethod
    def create(
        cls,
        name: str,
        email: str,
        password: Optional[str],
        provider: AuthProvider,
        token_version: int = 1,
        status: Optional["UserStatus"] = UserStatus.active,
        subscriptions: Optional[List["SubscriptionEntity"]] = None,
        dataset: Optional[List["DatasetEntity"]] = None,
        is_validated: Optional[bool] = False,
        role: Optional[Role] = Role.user,
    ) -> "UserEntities":
        return cls(
            name=name,
            email=email,
            status=status,
            password=password,
            token_version=token_version,
            is_validated=is_validated,
            role=role,
            provider=provider,
            subscriptions=subscriptions,
            dataset=dataset,
        )
