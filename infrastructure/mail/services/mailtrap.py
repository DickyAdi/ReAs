import aiosmtplib
from pydantic import EmailStr, validate_call

from domain.mail import SmtpInterface
from config.settings import settings

CREDS = {
    "smtp_host": settings.smtp_host,
    "smtp_username": settings.smtp_username,
    "smtp_password": settings.smtp_password,
}


class MailtrapService(SmtpInterface):
    """Infrastructure layer for mailing service using Mailtrap as the SMTP.

    Args:
        SmtpInterface (Class): Domain contract.
    """

    @validate_call
    async def send(
        self,
        email,
        sender: EmailStr,
        recepients: list[EmailStr] | EmailStr,
        smtp_credentials: dict = CREDS,
    ):
        """Send an email using Mailtrap SMTP.

        Args:
            email (Any): Email that want to be send.
            sender (EmailStr): Sender email.
            recepients (list[EmailStr] | EmailStr): Recepients.
            smtp_credentials (dict, optional): SMTP credentials. Defaults to CREDS.

        Returns:
            Any: aiosmtplib response.
        """
        return await aiosmtplib.send(
            email,
            sender=sender,
            recipients=recepients,
            hostname=smtp_credentials.get("smtp_host"),
            username=smtp_credentials.get("smtp_username"),
            password=smtp_credentials.get("smtp_password"),
            use_tls=False,
            start_tls=True,
        )
