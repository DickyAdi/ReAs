from application.mailing import MailingApplication
from infrastructure.mail import MailtrapService


def get_mailtrap_application() -> MailingApplication:
    """Get mailtrap application via dependency injection

    Returns:
        MailingApplication: Mailing application
    """
    return MailingApplication(smtp_provider=MailtrapService())
