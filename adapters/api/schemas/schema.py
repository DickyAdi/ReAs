from pydantic import BaseModel, EmailStr, Field

class RegisterRequest(BaseModel):
    name:str = Field(max_length=64)
    email:EmailStr
    password:str = Field(min_length=6, max_length=64, pattern=r'.*\d.*')

class Token(BaseModel):
    access_token:str
    token_type:str

class TokenData(BaseModel):
    email: str | None = None