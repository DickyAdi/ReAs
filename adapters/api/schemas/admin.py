from uuid import UUID
from typing import Optional
from pydantic import BaseModel, EmailStr, Field, field_validator
from decimal import Decimal
from domain.enums.users import Role
from domain.enums.tiers_pricing import Currency
from domain.enums.subscriptions import SubscriptionPlan
from domain.enums.tiers import Tier


class CreateUserRequest(BaseModel):
    name: str = Field(max_length=64)
    email: EmailStr
    password: str = Field(min_length=6, max_length=64, pattern=r".*\d.*")
    idempotency_key: UUID
    is_validated: bool


class CreateAdminRequest(BaseModel):
    name: str = Field(max_length=64)
    email: EmailStr
    password: str = Field(min_length=6, max_length=64, pattern=r".*\d.*")
    idempotency_key: UUID
    is_validated: bool


class GetUsersResponse(BaseModel):
    total: int
    offset: int
    limit: int
    data: list


class GetMeResponse(BaseModel):
    name: str
    email: EmailStr
    role: Role


class CreatePlanRequestForm(BaseModel):
    tier: Tier
    cycle: SubscriptionPlan
    currency: Currency
    price: Decimal
    duration_days: Optional[int]

    @field_validator("duration_days", mode="before")
    def empty_str_to_none(cls, v):
        if v in ("None", "null", "0", None, ""):
            return None
        return v
