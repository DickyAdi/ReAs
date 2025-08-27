from datetime import datetime, timedelta, timezone

from domain.entities.user.interfaces import UserInterface
from domain.auth.interfaces import AuthInterface
from domain.mail.smtp import SmtpInterface
from domain.exceptions import UserNotFoundError, InvalidResetPasswordTokenError
from domain.enums.auth import TokenPurpose

from application.mailing import MailingApplication

from config.settings import settings

from .base_class import BaseAuthenticationFlow


class ForgetPasswordFlow(BaseAuthenticationFlow):
    """Use case flow for forget password.

    Args:
        BaseAuthenticationFlow (Class): Base class for the authentication flow.
    """

    def __init__(
        self,
        user_service: UserInterface,
        auth_service: AuthInterface,
        mail_service: SmtpInterface,
    ):
        super().__init__(user_service=user_service, auth_service=auth_service)
        self.mail_app = MailingApplication(smtp_provider=mail_service)

    async def __call__(self, db, user_agent, email: str):
        """Flow for forget password. This function will validate user credentials and will send forget password link to users email.

        Args:
            db (Any): Infrastructure layer database session.
            user_agent (_type_): User agent of the requester.
            email (str): Users email.

        Raises:
            UserNotFoundException: If there's no match users email with the given `email`.

        Returns:
            bool: Whether email has been sent or not.
        """
        user = await self.user_app.get_user(db=db, email=email)
        if not user:
            raise UserNotFoundError(user_identifier=email)
        token = self.auth_app.create_token(
            email=email,
            token_version=1,
            expire_time=timedelta(minutes=15),
            purpose=TokenPurpose.forget_password,
        )
        time_sent = datetime.now(timezone.utc)
        reset_link = f"{settings.domain}reset-password?token={token}"
        email_sent = await self.mail_app.send_reset_password_email(
            user_agent=user_agent,
            request_time=time_sent,
            reset_link=reset_link,
            sender="support@reas.com",
            recepients=email,
        )
        if email_sent:
            return True
        return False


class ResetPasswordFlow(BaseAuthenticationFlow):
    """Use case for resetting users password.

    Args:
        BaseAuthenticationFlow (Class): Base class for the authentication flow.
    """

    def __init__(self, user_service: UserInterface, auth_service: AuthInterface):
        super().__init__(user_service=user_service, auth_service=auth_service)

    async def __call__(self, db, token, password):
        """Flow for resetting users password. This function must be used to decode the given token from `ForgetPasswordFlow` Class.

        Args:
            db (Any): Infrastructure layer database session.
            token (Str): Forget password token.
            password (Str): Users new password.

        Raises:
            InvalidResetPasswordToken: If the given `token` is invalid.
            UserNotFoundException: If user is not found on the database.

        Returns:
            bool: Whether password has been edited or not.
        """
        payload = self.auth_app.decode_token(token=token)
        email = payload.get("sub")
        purpose = payload.get("purpose")
        token_version = payload.get("token_version")
        hashed_password = self.auth_app.hash_password(value=password)
        if (
            email is None
            or purpose != TokenPurpose.forget_password.name
            or token_version != 1
        ):
            raise InvalidResetPasswordTokenError
        user = await self.user_app.get_user(db=db, email=email)
        if not user:
            raise UserNotFoundError(user_identifier=email)
        is_edited = await self.user_app.change_user_password(
            db=db, user=user, hashed_changed_password_value=hashed_password
        )
        if is_edited:
            return True
        return False
