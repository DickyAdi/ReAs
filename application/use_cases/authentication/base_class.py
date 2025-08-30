# from domain.exceptions import UserNotFoundException

from domain.entities.user.interfaces import UserInterface
from domain.auth.interfaces import AuthInterface
from domain.exceptions import InvalidCredentialsError
from domain.enums.users import Role

from application.user import UserApplication
from application.auth import AuthApplication


class BaseAuthenticationFlow:
    """Base class to be inherited for use cases that need authentication and user services."""

    def __init__(self, user_service: UserInterface, auth_service: AuthInterface):
        self.user_app = UserApplication(service=user_service)
        self.auth_app = AuthApplication(service=auth_service)

    async def verify_credential(
        self, db, email: str, password: str, role: Role = Role.user
    ):
        """Verify user input credentials.

        Args:
            db (Any): Session of the database.
            email (str): User input email.
            password (str): User input password.
            role (Role, optional): Filter users by role. Defaults to Role.user. If None, no filter will be applied.

        Raises:
            InvalidCredentialsError: If credentials is not valid.

        Returns:
            User: Infrastructure layer User ORM instances.
        """
        user = await self.user_app.get_user(db=db, email=email, role=role)

        if user:
            valid_password = self.auth_app.verify_password(
                user_password=password, db_password=user.password
            )
        else:  # * Even if user doesnt exist, keep verifying a password by using a dummy password to prevent timing attacks `ref: CWE-208 Observable timing discrepancy`
            valid_password = self.auth_app.verify_password(
                user_password=password,
                db_password="$2a$12$YME8U1LNU2PMQRBCnOMhYOVyRhtTYzol7EJEPOX2imhEpnCfOkSbe",
            )
            valid_password = False
        if not user or not valid_password:
            raise InvalidCredentialsError("email/password")
        return user
