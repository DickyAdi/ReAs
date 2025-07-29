from fastapi import APIRouter, status, Request, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Annotated

from infrastructure.db import get_db
from infrastructure.db.services import UserService
from infrastructure.auth import AuthService
from application.user import UserApplication
from ...schemas import Token, TokenData

router = APIRouter(prefix='/auth', tags=['auth'])

@router.post('/token')
async def login(form_data:Annotated[OAuth2PasswordRequestForm, Depends()], db:AsyncSession=Depends(get_db)) -> Token:
    flow = UserApplication(UserService(), AuthService())
    try:
        token = await flow.login_for_access_token(db=db, email=form_data.username, password=form_data.password)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={'WWW-Authenticate' : "Bearer"}
        )
    return Token(**token.__dict__)
