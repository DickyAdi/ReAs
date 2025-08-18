from datetime import timedelta

from domain.auth.interfaces import AuthInterface
from domain.entities.user.interfaces import UserInterface
from domain.auth import Token
from domain.enums.auth import TokenPurpose
from domain.enums.users import Role
from domain.exceptions import InvalidCredentials, InvalidAuthenticateToken
from .base_class import BaseAuthenticationFlow


class UserLoginFlow(BaseAuthenticationFlow):
    """Use case flow for user login, this includes admin login flow.

    Args:
        BaseAuthenticationFlow (Class): Base authentication flow.
    """

    def __init__(self, user_service: UserInterface, auth_service: AuthInterface):
        super().__init__(auth_service=auth_service, user_service=user_service)

    async def __call__(
        self,
        db,
        email: str,
        password: str,
        expired_timedelta: timedelta = timedelta(days=1),
        role: Role = Role.user,
    ):
        """This function is for authenticating user credentials and creating a JWT for their auth purposes.

        Args:
            db (Any): Infrastructure layer database session.
            email (str): Users email input.
            password (str): Users password input.
            expired_timedelta (timedelta, optional): JWT expiration time. Defaults to timedelta(days=1).
            role (Role, optional): Filter users role for credential verification. Defaults to Role.user.

        Returns:
            Token: Token instance.
        """
        user = await self.verify_credential(
            db=db, email=email, password=password, role=role
        )
        access_token = self.auth_app.create_token(
            email=email,
            token_version=user.token_version,
            expire_time=expired_timedelta,
            purpose=TokenPurpose.authenticate,
        )
        return Token(access_token=access_token, token_type="bearer")


class UserAuthenticateFlow(BaseAuthenticationFlow):
    """Use case flow for user authentication, this includes admin authentication flow.

    Args:
        BaseAuthenticationFlow (Class): Base authentication flow.
    """

    def __init__(self, user_service: UserInterface, auth_service: AuthInterface):
        super().__init__(auth_service=auth_service, user_service=user_service)

    async def __call__(self, db, token, with_sub: bool = False):
        """Decode authenticate token purpose.

        Args:
            db (Any): Infrastructure layer database session.
            token (str): Users JWT.
            with_sub (bool, optional): Whether to join related subscriptions or not. Defaults to False.

        Raises:
            InvalidCredentials: If the required key in the JWT is missing, missmatch token purpose, or Exception.
            InvalidAuthenticateToken: If users token version isnt match.

        Returns:
            User: Infrastructure layer Users ORM instance.
        """
        try:
            payload = self.auth_app.decode_token(token=token)
            email = payload.get("sub")
            if email is None:
                raise InvalidCredentials
            token_version = payload.get("token_version")
            if token_version is None:
                raise InvalidCredentials
            purpose = payload.get("purpose")
            if purpose is None or purpose != TokenPurpose.authenticate.name:
                raise InvalidCredentials
        except Exception as e:
            raise InvalidCredentials(f"Something went wrong: {str(e)}")
        user = await self.user_app.get_user(
            db=db, email=email, with_sub=with_sub, role=None
        )  # * Role=None means there's no role checking, later if want to be stricter then this should be set accordingly based on the expected user roles
        if user.token_version != token_version:
            raise InvalidAuthenticateToken("Token version wasnt match.")
        return user


# !This 2 class below arent used.
# TODO: Refactor this to remove all import path in the other module/package
class AdminLoginFlow(BaseAuthenticationFlow):
    def __init__(self, user_service: UserInterface, auth_service: AuthInterface):
        super().__init__(auth_service=auth_service, user_service=user_service)

    async def __call__(
        self,
        db,
        email: str,
        password: str,
        expired_timedelta: timedelta = timedelta(days=1),
    ):
        user = await self.verify_credential(
            db=db, email=email, password=password, role=Role.admin
        )
        access_token = self.auth_app.create_token(
            email=email,
            token_version=user.token_version,
            expire_time=expired_timedelta,
            purpose=TokenPurpose.authenticate,
        )
        return Token(access_token=access_token, token_type="bearer")


class SuperadminLoginFlow(BaseAuthenticationFlow):
    def __init__(self, user_service: UserInterface, auth_service: AuthInterface):
        super().__init__(auth_service=auth_service, user_service=user_service)

    async def __call__(
        self,
        db,
        email: str,
        password: str,
        expired_timedelta: timedelta = timedelta(days=1),
    ):
        user = await self.verify_credential(
            db=db, email=email, password=password, role=Role.superadmin
        )
        access_token = self.auth_app.create_token(
            email=email,
            token_version=user.token_version,
            expire_time=expired_timedelta,
            purpose=TokenPurpose.authenticate,
        )
        return Token(access_token=access_token, token_type="bearer")
