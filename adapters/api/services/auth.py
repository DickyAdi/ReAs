from fastapi import Depends
from fastapi.security import OAuth2PasswordBearer
from typing import Annotated
from sqlalchemy.ext.asyncio import AsyncSession

from infrastructure.db import get_db
from infrastructure.auth import AuthService
from infrastructure.db.services import UserService
from infrastructure.db import User

oauth2_scheme = OAuth2PasswordBearer('/api/auth/token')

async def get_current_user(token:Annotated[str, Depends(oauth2_scheme)], db:AsyncSession=Depends(get_db)):
    user = await AuthService().decode_user_token(token=token, db=db, get_user_email=UserService().get_user_by_email)
    return user


async def get_current_active_user(current_user:User = Depends(get_current_user)):
    return current_user