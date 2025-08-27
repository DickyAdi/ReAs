import bcrypt
from sqlalchemy.ext.asyncio import AsyncSession
import jwt
from jwt.exceptions import InvalidTokenError, ExpiredSignatureError

from domain.auth import AuthInterface
from config.settings import settings
from domain.auth import TokenData
from domain.exceptions import InvalidAuthenticateTokenError, InvalidCredentialsError


class AuthService(AuthInterface):
    def verify_password(self, user_password: str, db_password: str):
        return bcrypt.checkpw(
            user_password.encode("utf-8"), db_password.encode("utf-8")
        )

    def create_access_token(self, data: dict):
        encoded_token = jwt.encode(
            data, settings.jwt_secret_key, settings.jwt_algorithm
        )
        return encoded_token

    async def decode_user_token(self, token, db: AsyncSession, get_user_email):
        try:
            payload = self.decode_token(token=token)
            email = payload.get("sub")
            token_version = payload.get("token_version")
            if email is None:
                raise InvalidAuthenticateTokenError
            token_data = TokenData(email=email)
        except InvalidTokenError:
            raise InvalidAuthenticateTokenError
        user = await get_user_email(db=db, email=token_data.email)
        if user is None:
            raise InvalidAuthenticateTokenError
        if user.token_version != token_version:
            raise InvalidCredentialsError(credentials_type="token_version")
        return user

    def decode_token(self, token):
        try:
            payload = jwt.decode(
                token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm]
            )
            return payload
        except ExpiredSignatureError:
            raise InvalidAuthenticateTokenError(message="Token has expired.")

    def hash(self, value: str):
        return bcrypt.hashpw(value.encode("utf-8"), salt=bcrypt.gensalt()).decode(
            "utf-8"
        )
