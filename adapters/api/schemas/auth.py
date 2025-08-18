from pydantic import BaseModel, EmailStr, Field

class ForgetPasswordRequest(BaseModel):
    email:EmailStr

class ResetPasswordRequest(BaseModel):
    password:str = Field(min_length=6, max_length=64, pattern=r'.*\d.*')