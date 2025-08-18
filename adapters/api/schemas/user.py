from uuid import UUID
from pydantic import BaseModel, EmailStr, ConfigDict, Field
from domain.enums.users import Role
from domain.enums.tiers import Tier

class RegisterRequest(BaseModel):
    name:str = Field(max_length=64)
    email:EmailStr
    password:str = Field(min_length=6, max_length=64, pattern=r'.*\d.*')
    idempotency_key:UUID

class GetMeResponse(BaseModel):
    name:str
    email:EmailStr
    role:Role
    subscription_id:UUID

class EditMeRequest(BaseModel):
    name:str
    model_config = ConfigDict(extra="forbid")

class ChangeEmailRequest(BaseModel):
    email:EmailStr

class ChangePasswordRequest(BaseModel):
    password:str