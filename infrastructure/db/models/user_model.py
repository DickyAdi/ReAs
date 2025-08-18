from sqlalchemy import Integer, String, DateTime
from uuid import UUID, uuid4
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.dialects.postgresql import ENUM
from datetime import datetime, timezone
import bcrypt
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from domain.entities.user import UserEntities
    from .subscriptions_model import Subscriptions
    from .scrape_usages_model import ScrapeUsages


from domain.enums.users import AuthProvider, Role, UserStatus
from ..db import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, autoincrement=True, primary_key=True, index=True, nullable=False)
    uuid: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), default=uuid4, unique=True, index=True, nullable=False)
    name: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    password: Mapped[str] = mapped_column(String(64))
    token_version:Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    is_validated: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[DateTime] = mapped_column(DateTime, default=datetime.now(timezone.utc), nullable=False)
    updated_at: Mapped[DateTime] = mapped_column(DateTime, default=datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc), nullable=False)
    role: Mapped[Role] = mapped_column(ENUM(Role, name='user_role_enum', create_type=True, check_first=True), default=Role.user, nullable=False)
    status: Mapped[UserStatus] = mapped_column(ENUM(UserStatus, name="user_status_enum", create_type=True, check_first=True), nullable=False, default=UserStatus.active)
    provider: Mapped[AuthProvider] = mapped_column(ENUM(AuthProvider, name="user_auth_provider_enum", create_type=True, check_first=True), nullable=False)

    subscriptions:Mapped[List['Subscriptions']] = relationship("Subscriptions", back_populates='user')
    scrape_usages:Mapped[List['ScrapeUsages']] = relationship("ScrapeUsages", back_populates='user')

    def to_entity(self) -> 'UserEntities':
        return UserEntities(
            id = self.id,
            uuid = self.uuid,
            name = self.name,
            email  = self.email,
            password = self.password,
            token_version= self.token_version,
            is_validated = self.is_validated,
            created_at = self.created_at,
            updated_at = self.updated_at,
            role = self.role,
            status = self.status,
            provider=self.provider,
        )
    @classmethod
    def from_entity(cls, user:'UserEntities') -> "User":
        return cls(
            name=user.name,
            email=user.email,
            password=bcrypt.hashpw(bytes(user.password.encode('utf-8')), bcrypt.gensalt()).decode('utf-8') if user.password else None,
            is_validated=user.is_validated,
            role=user.role,
            provider=user.provider,
            token_version=user.token_version
        )