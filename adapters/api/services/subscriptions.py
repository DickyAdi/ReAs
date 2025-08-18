from application.subscriptions import (
    SubscriptionsApplication,
    SubscriptionPlanApplication,
)
from application.transactions import TransactionApplication
from application.use_cases.subscription import CreateNewPlanFlow

from infrastructure.db.services import (
    SubscriptionService,
    SubscriptionPlanService,
    TransactionService,
    PaymentService,
    TierService,
)


def get_subscription_application() -> SubscriptionsApplication:
    """Get subscription application via dependency injection.

    Returns:
        SubscriptionsApplication: Subscription application.
    """
    return SubscriptionsApplication(subscription_service=SubscriptionService())


def get_transaction_application() -> TransactionApplication:
    """Get transaction application via dependency injection.

    Returns:
        TransactionApplication: Transaction application.
    """
    return TransactionApplication(transaction_service=TransactionService())


def get_subscription_plan_application() -> SubscriptionPlanApplication:
    """Get subscription plan/plan application via dependency injection.

    Returns:
        SubscriptionPlanApplication: Subscription plan application.
    """
    return SubscriptionPlanApplication(service=SubscriptionPlanService())


def get_create_plan_flow() -> CreateNewPlanFlow:
    """Get create/append new plan flow from application use cases.

    Returns:
        CreateNewPlanFlow: Create new plan application use cases flow.
    """
    return CreateNewPlanFlow(
        tier_service=TierService(),
        subscription_service=SubscriptionService(),
        plan_service=SubscriptionPlanService(),
        payment_service=PaymentService(),
        transaction_service=TransactionService(),
    )
