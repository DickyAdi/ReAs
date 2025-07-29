from sqlalchemy import Integer, String, DateTime, Enum
from uuid import UUID, uuid4
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from datetime import datetime, timezone
import bcrypt

from domain.entities.user import UserEntities
from domain.tier import Tier
from domain.provider import Provider
from ..db import Base

class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, autoincrement=True, primary_key=True, index=True)
    uuid: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), default=uuid4, unique=True, index=True, nullable=False)
    name: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    password: Mapped[str] = mapped_column(String(64))
    is_validated: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[DateTime] = mapped_column(DateTime, default=datetime.now(timezone.utc), nullable=False)
    updated_at: Mapped[DateTime] = mapped_column(DateTime, default=datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc))
    is_superuser: Mapped[bool] = mapped_column(default=False, nullable=False)
    tier: Mapped[Tier] = mapped_column(Enum(Tier, name="user_tier_enum"), default=Tier.Base, nullable=False)
    provider: Mapped[Provider] = mapped_column(Enum(Provider, name="user_auth_provider_enum"), nullable=False)

    def to_entity(self) -> UserEntities:
        return UserEntities(
            id = self.id,
            uuid = self.uuid,
            name = self.name,
            email  = self.email,
            password = self.password,
            is_validated = self.is_validated,
            created_at = self.created_at,
            updated_at = self.updated_at,
            is_superuser = self.is_superuser,
            tier=self.tier,
            provider=self.provider
        )
    @classmethod
    def from_entity(cls, user:UserEntities) -> "User":
        return cls(
            name=user.name,
            email=user.email,
            password=bcrypt.hashpw(bytes(user.password.encode('utf-8')), bcrypt.gensalt()).decode('utf-8') if user.password else None,
            is_validated=user.is_validated,
            is_superuser=user.is_superuser,
            tier=user.tier,
            provider=user.provider
        )