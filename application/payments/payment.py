from typing import Optional
from decimal import Decimal

from domain.entities.payments.interfaces import PaymentInterface
from domain.entities.payments import PaymentEntity

from domain.enums.payments import PaymentMethod, PaymentStatus


class PaymentApplication:
    """Payment application for payment services."""

    def __init__(self, service: PaymentInterface):
        self.service = service

    async def issue_payment(
        self,
        db,
        method: PaymentMethod,
        amount_charged: Decimal,
        status: Optional[PaymentStatus] = PaymentStatus.pending,
    ):
        """Issue new payment

        Args:
            db (AsyncSession): Infrastructure DB async session.
            method (PaymentMethod): PaymentMethod Enums for the method used for they payment
            amount_charged (Decimal): The amount that would be charged to the user
            status (Optional[PaymentStatus], optional): Status of the payment. Defaults to PaymentStatus.pending.

        Returns:
            Payment Obj: Infrastructure layer payment object.
        """
        payment_obj = PaymentEntity.create(
            method=method, amount_charged=amount_charged, status=status
        )
        payment = await self.service.create_payment(
            db=db, payment=payment_obj, commit=False
        )
        return payment
