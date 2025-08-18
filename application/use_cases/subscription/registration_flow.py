from typing import Optional
from uuid import UUID

from domain.entities.subscriptions.interfaces import (
    SubscriptionInterface,
    SubscriptionPlansInterface,
)
from domain.entities.payments.interfaces import PaymentInterface
from domain.entities.transactions.interfaces import TransactionInterface
from domain.entities.user.interfaces import UserInterface
from domain.entities.user import UserEntities
from domain.enums.users import AuthProvider, Role
from domain.enums.tiers import Tier
from domain.enums.subscriptions import SubscriptionPlan, SubscriptionStatus
from domain.enums.tiers_pricing import Currency
from domain.enums.payments import PaymentMethod, PaymentStatus
from domain.enums.transactions import TransactionStatus

from .base_class import BaseSubscriptionFlow


class UserRegistrationFlow(BaseSubscriptionFlow):
    """Use case for user registration flow.

    Args:
        BaseSubscriptionFlow (Class): Base subcriptions class use case.
    """

    def __init__(
        self,
        user_service: UserInterface,
        subscription_service: SubscriptionInterface,
        plan_service: SubscriptionPlansInterface,
        transaction_service: TransactionInterface,
        payment_service: PaymentInterface,
    ):
        super().__init__(
            payment_service=payment_service,
            subscription_service=subscription_service,
            plan_service=plan_service,
            transaction_service=transaction_service,
        )
        self.user_service = user_service

    async def __call__(
        self,
        db,
        name: str,
        email: str,
        password: str,
        provider: AuthProvider,
        idempotency_key: UUID,
        is_validated: Optional[bool] = False,
        role: Optional[Role] = Role.user,
        commit: bool = True,
    ):
        """User registration flow, including creating new subscriptions, transaction, and payment intent which automatically be active, paid, and success.

        Args:
            db (Any): Infrastructure layer database session.
            name (str): Users name. Not Username.
            email (str): Users email.
            password (str): Users password.
            provider (AuthProvider): Authentication provider. This refers to who will authenticate the user, if social login was used, this parameter must be filled with the used social platform, or if user is directly registering to the app without any social registration, then this parameter must be filled with AuthProvider.local.
            idempotency_key (UUID): Idempotency key used for transaction and payment.
            is_validated (Optional[bool], optional): Users email validation. Defaults to False.
            role (Optional[Role], optional): Users role. Defaults to Role.user. NEVER CHANGE THIS UNLESS YOU KNOW WHAT YOU ARE DOING!
            commit (bool, optional): Whether to commit to database or not. Defaults to True. (This was added for seeding strategy, wont expect this thing to change too much in the logic)

        Returns:
            User: Infrastructure layer User ORM instance.
        """
        new_user = UserEntities.create(
            name=name,
            email=email,
            password=password,
            is_validated=is_validated,
            role=role,
            provider=provider,
        )
        created_user = await self.user_service.create_user(
            db=db, user=new_user, commit=False
        )
        new_subs = await self.issue_subscription(  # noqa
            db=db,
            tier=Tier.Base,
            cycle=SubscriptionPlan.free,
            currency=Currency.idr,
            user_id=created_user.id,
            idempotency_key=idempotency_key,
            method=PaymentMethod.local,
            payment_status=PaymentStatus.success,
            trx_status=TransactionStatus.paid,
            subs_status=SubscriptionStatus.active,
        )
        if commit:
            await db.commit()
        return created_user
