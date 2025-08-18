from application.user import UserApplication
from application.use_cases.subscription import UserRegistrationFlow
from application.use_cases.authentication import (
    SendVerifyEmailFlow,
    UserLoginFlow,
    VerifyEmailFlow,
    AdminLoginFlow,
    SuperadminLoginFlow,
)
from infrastructure.db.services import (
    UserService,
    PaymentService,
    TransactionService,
    SubscriptionPlanService,
    SubscriptionService,
)
from infrastructure.auth import AuthService
from infrastructure.mail.services import MailtrapService


def get_registration_flow():
    """Get application registration use cases flow via dependency injection.

    Returns:
        UserRegistrationFlow: User registration flow class object
    """
    return UserRegistrationFlow(
        payment_service=PaymentService(),
        user_service=UserService(),
        plan_service=SubscriptionPlanService(),
        subscription_service=SubscriptionService(),
        transaction_service=TransactionService(),
    )


def get_user_application():
    """Get user application via dependency injection.

    Returns:
        UserApplication: User application class object.
    """
    return UserApplication(service=UserService())


def get_login_flow():
    """Get application login use cases flow via dependency injection.

    Returns:
        UserLoginFlow: Login flow class object.
    """
    return UserLoginFlow(user_service=UserService(), auth_service=AuthService())


def get_admin_login_flow():
    """Get application login use cases flow for admin role via dependency injection.

    Returns:
        AdminLoginFlow: Admin login flow class object.
    """
    return AdminLoginFlow(user_service=UserService(), auth_service=AuthService())


def get_superadmin_login_flow():
    """Get application login use cases flow for superadmin role via dependency injection.

    Returns:
        SuperAdminLoginFlow: Superadmin login flow class object.
    """
    return SuperadminLoginFlow(user_service=UserService(), auth_service=AuthService())


def get_verify_email_flow():
    """Get application use cases for verify email flow via dependency injection.

    Returns:
        VerivyEmailFlow: Verify email flow class object instance.
    """
    return VerifyEmailFlow(auth_service=AuthService(), user_service=UserService())


def get_send_verify_email_flow():
    """Get application use cases for sending verify email flow via dependency injection.

    Returns:
        SendVerifyEmailFlow: Send verify email flow class object instance.
    """
    return SendVerifyEmailFlow(
        auth_service=AuthService(),
        user_service=UserService(),
        mail_service=MailtrapService(),
    )
