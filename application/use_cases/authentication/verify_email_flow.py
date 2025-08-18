from datetime import timedelta

from domain.auth.interfaces import AuthInterface
from domain.entities.user.interfaces import UserInterface
from domain.mail.smtp import SmtpInterface
from domain.exceptions import (
    UserNotFoundException,
    EmailAlreadyValidated,
    InvalidEmailVerifyToken,
)
from domain.enums.auth import TokenPurpose

from application.mailing import MailingApplication
from config.settings import settings

from .base_class import BaseAuthenticationFlow


class SendVerifyEmailFlow(BaseAuthenticationFlow):
    """Use case flow for sending verify users email.

    Args:
        BaseAuthenticationFlow (Class): Base class for authentication flow.
    """

    def __init__(
        self,
        auth_service: AuthInterface,
        user_service: UserInterface,
        mail_service: SmtpInterface,
    ):
        super().__init__(user_service=user_service, auth_service=auth_service)
        self.mail_app = MailingApplication(smtp_provider=mail_service)

    async def __call__(self, user):
        """Send email verification for authenticated users.

        Args:
            user (Any): Infrastructure layer User ORM instance.

        Raises:
            UserNotFoundException: If `user` is None.
            EmailAlreadyValidated: If user is already validated.
            Exception: If email is not sent. This is likely to be unrecognizeable Exception within the Mailiing Application.

        Returns:
            bool: Whether email has been sent or not.
        """
        if not user:
            raise UserNotFoundException(message="`user` cannot be None")
        if user.is_validated:
            raise EmailAlreadyValidated("Email is already validated.")
        token = self.auth_app.create_token(
            email=user.email,
            token_version=user.token_version,
            expire_time=timedelta(minutes=15),
            purpose=TokenPurpose.verify_email,
        )
        verify_link = f"{settings.domain}/verify-email-token?token={token}"
        sended = await self.mail_app.send_verify_email(
            verify_link=verify_link, sender="support@reas.org", recepients=user.email
        )
        if not sended:
            raise Exception("Something went wrong.")
        return sended


class VerifyEmailFlow(BaseAuthenticationFlow):
    """Use case flow for verifying users email based on verify email token.

    Args:
        BaseAuthenticationFlow (Class): Base class for authentication flow.
    """

    def __init__(self, auth_service: AuthInterface, user_service: UserInterface):
        super().__init__(user_service=user_service, auth_service=auth_service)

    async def __call__(self, token, db):
        """Flow for verifying users verify email token and verifying users email.

        Args:
            token (str): Users verify email token, received via query parameter in the link sent to the users email.
            db (Any): Infrastructure layer database session.

        Raises:
            InvalidEmailVerifyToken: If given `token` purpose is not TokenPurpose.verify_email
            UserNotFoundException: If User is not on database.

        Returns:
            bool: Whether users email has been validated or not.
        """
        payload = self.auth_app.decode_token(token=token)
        email = payload.get("sub")
        purpose = payload.get("purpose")
        if email is None or purpose != TokenPurpose.verify_email.name:
            raise InvalidEmailVerifyToken
        user = await self.user_app.get_user(db=db, email=email)
        if not user:
            raise UserNotFoundException
        is_edited = self.user_app.set_user_validated(db=db, user=user)
        return is_edited
