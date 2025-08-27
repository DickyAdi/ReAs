from email.message import EmailMessage

from domain.mail import SmtpInterface
from domain.mail import ResetPasswordTemplate, VerifyEmailTemplate


class MailingApplication:
    """Mailing application for mailing services"""

    def __init__(self, smtp_provider: SmtpInterface):
        self.provider: SmtpInterface = smtp_provider

    async def send_verify_email(self, verify_link: str, sender: str, recepients: str):
        """Send verify email method.

        Args:
            verify_link (str): Verify email link with token as Query parameter.
            sender (str): Sender email.
            recepients (str): Recipients email.

        Returns:
            bool: Whether the email has been sent or not
        """
        template = VerifyEmailTemplate()
        content = template(verify_link=verify_link)
        msg = EmailMessage()
        msg["Subject"] = "Verify email for ReAs App"
        msg["From"] = sender
        msg["To"] = recepients
        msg.set_content(f"Verify your email account at {verify_link}")
        msg.add_alternative(content, subtype="html")
        try:  # defensive. Still have no idea what are the exceptions for SMTP's. Bare raise everything.
            await self.provider.send(email=msg, sender=sender, recepients=recepients)
            return True
        except Exception:
            raise

    async def send_reset_password_email(
        self, user_agent, request_time, reset_link: str, sender: str, recepients: str
    ):
        """Send reset password email.

        Args:
            user_agent (str): Requester user agent.
            request_time (str): Requester request time.
            reset_link (str): Reset link for reset password.
            sender (str): Sender email.
            recepients (str): Recipient email

        Returns:
            bool: Whether the email has been sent or not.
        """
        template = ResetPasswordTemplate()
        content = template(
            device=user_agent, request_time=request_time, reset_link=reset_link
        )
        msg = EmailMessage()
        msg["Subject"] = "Reset password for ReAs App"
        msg["From"] = sender
        msg["To"] = recepients
        msg.set_content(f"Reset your account password at {reset_link}")
        msg.add_alternative(content, subtype="html")
        try:  # defensive. Still have no idea what are the exceptions for SMTP's. Bare raise everything.
            await self.provider.send(email=msg, sender=sender, recepients=recepients)
            return True
        except Exception:
            raise
