import bcrypt
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import timedelta, datetime, timezone
import jwt
from jwt.exceptions import InvalidTokenError
from fastapi import HTTPException, status

from domain.auth import AuthInterface
from domain.entities.user import UserEntities
from infrastructure.db.models import User
from config.settings import settings
from domain.auth import Token, TokenData

class AuthService(AuthInterface):
    def verify_password(self, user_password:str, db_password:str):
        return bcrypt.checkpw(user_password.encode('utf-8'), db_password.encode('utf-8'))
    # async def authenticate_user(self, email:str, password:str, db:AsyncSession):
    def create_access_token(self, data:dict, expires_time:timedelta):
        to_encode = data.copy()
        exp = datetime.now(timezone.utc) + expires_time
        to_encode.update({'exp' : exp})
        encoded_token = jwt.encode(to_encode, settings.jwt_secret_key, settings.jwt_algorithm)
        return encoded_token
    async def decode_user_token(self, token, db, get_user_email):
        credentials_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials.",
            headers={"WWW-Authenticate" : "Bearer"}
        )
        try:
            payload = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
            email = payload.get('sub')
            if email is None:
                raise credentials_exception
            token_data = TokenData(email=email)
        except InvalidTokenError:
            raise credentials_exception
        user = await get_user_email(db=db, email=token_data.email)
        if user is None:
            raise credentials_exception
        return user