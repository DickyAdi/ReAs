from typing import Optional
from uuid import UUID

from domain.entities.subscriptions.interfaces import (
    SubscriptionInterface,
    SubscriptionPlansInterface,
)
from domain.entities.payments.interfaces import PaymentInterface
from domain.entities.transactions.interfaces import TransactionInterface

from domain.enums.payments import PaymentMethod, PaymentStatus
from domain.enums.transactions import TransactionStatus
from domain.enums.subscriptions import SubscriptionPlan, SubscriptionStatus
from domain.enums.tiers import Tier
from domain.enums.tiers_pricing import Currency

from application.subscriptions import (
    SubscriptionsApplication,
    SubscriptionPlanApplication,
)
from application.transactions import TransactionApplication
from application.payments import PaymentApplication


class BaseSubscriptionFlow:
    """Base class to be inherited for any subscriptions use cases."""

    def __init__(
        self,
        subscription_service: SubscriptionInterface,
        plan_service: SubscriptionPlansInterface,
        transaction_service: TransactionInterface,
        payment_service: PaymentInterface,
    ):
        self.payment_app = PaymentApplication(service=payment_service)
        self.subscription_app = SubscriptionsApplication(
            subscription_service=subscription_service
        )
        self.plan_app = SubscriptionPlanApplication(service=plan_service)
        self.transaction_app = TransactionApplication(
            transaction_service=transaction_service
        )

    async def issue_subscription(
        self,
        db,
        tier: Tier,
        cycle: SubscriptionPlan,
        currency: Currency,
        user_id: int,
        idempotency_key: UUID,
        method: PaymentMethod,
        payment_status: Optional[PaymentStatus] = PaymentStatus.pending,
        trx_status: Optional[TransactionStatus] = TransactionStatus.pending,
        subs_status: Optional[SubscriptionStatus] = SubscriptionStatus.pending,
    ):
        """Issue a new subscription, transaction, and payment intent for user. Including when user is registering, subscribing to a plan, or downgrading user subscription plan.

        Args:
            db (Any): Infrastructure layer database session.
            tier (Tier): To be subscribed tier.
            cycle (SubscriptionPlan): To be subscribed plan billing cycle.
            currency (Currency): To be subscribed plan currency.
            user_id (int): Users id for the new subscription.
            idempotency_key (UUID): Idempotency key for this flow.
            method (PaymentMethod): To be used payment method to pay the payment intent.
            payment_status (Optional[PaymentStatus], optional): Payment status for the payment intent. Defaults to PaymentStatus.pending. Directly change this to PaymentStatus.success only for registration flow, other than that DO NOT DIRECTLY CHANGE THIS PARAMETER.
            trx_status (Optional[TransactionStatus], optional): Transaction status for the transaction log in the database. Defaults to TransactionStatus.pending. Directly change this to TransactionStatus.paid only for registration flow, other than that DO NOT DIRECTLY CHANGE THIS PARAMETER.
            subs_status (Optional[SubscriptionStatus], optional): Subscription status for the user subscriptions. Defaults to SubscriptionStatus.pending. Directly change this to SubscriptionStatus.active for registration flow, other than that DO NOT DIRECTLY CHANGE THIS PARAMETER.

        Returns:
            Subsriptions: Infrastructure layer Subscriptions ORM instance.
        """
        plan = await self.plan_app.get_plan(
            db=db, tier=tier, cycle=cycle, currency=currency
        )
        payment_intent = await self.payment_app.issue_payment(
            db=db, method=method, amount_charged=plan.price, status=payment_status
        )
        plan_snapshot = {
            "code": plan.code,
            "price": plan.price,
            "currency": plan.currency.name,
            "created_at": plan.created_at,
        }
        trx = await self.transaction_app.issue_transaction(
            db=db,
            amount=plan.price,
            idempotency_key=idempotency_key,
            plan_snapshot=plan_snapshot,
            payment_id=payment_intent.id,
            trx_status=trx_status,
        )
        subs = await self.subscription_app.issue_subscription(
            db=db,
            user_id=user_id,
            end_subs=plan.duration_days,
            plan_id=plan.id,
            transaction_id=trx.id,
            subs_status=subs_status,
        )
        return subs
