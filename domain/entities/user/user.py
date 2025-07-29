from uuid import UUID
from dataclasses import dataclass
from typing import Optional
from datetime import datetime

from domain.tier import Tier
from domain.provider import Provider

@dataclass
class UserEntities:
    name:str
    email:str
    provider:Provider
    password:Optional[str | None]
    tier:Optional[Tier] = Tier.Base
    created_at:Optional[datetime | None] = None #derived from database
    updated_at:Optional[datetime | None] = None #derived from database
    id:Optional[int | None] = None #derived from database
    uuid:Optional[UUID | None] = None #derived from database
    is_validated:Optional[bool]=False
    is_superuser:Optional[bool]=False

    @classmethod
    def create(cls, name:str, email:str, password:str, provider:Provider, tier:Optional[Tier]=Tier.Base, is_validated:Optional[bool]=False, is_superuser:Optional[bool]=False) -> "UserEntities":
        return cls(
            name=name,
            email=email,
            password=password,
            tier=tier,
            is_validated=is_validated,
            is_superuser=is_superuser,
            provider=provider
        )