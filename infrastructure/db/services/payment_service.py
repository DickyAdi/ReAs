from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError
from typing import Optional

from ..models.payment_model import Payments
from ...db.error_mapper import DatabaseErrorMapper

from domain.entities.payments.interfaces import PaymentInterface
from domain.entities.payments import PaymentEntity


class PaymentService(PaymentInterface):
    async def create_payment(
        self, db: AsyncSession, payment: PaymentEntity, commit: Optional[bool] = True
    ):
        """Create new payment intent in the database.

        Args:
            db (AsyncSession): Database session.
            payment (PaymentEntity): Domain payment entity.
            commit (Optional[bool], optional): Whether to commit DML to the database or not. Defaults to True.

        Returns:
            Payments: New Payments ORM instance.
        """
        new_payment = Payments.from_entity(payment=payment)
        try:
            db.add(new_payment)
            await db.flush()
            if commit:
                await db.commit()
                await db.refresh(new_payment)
            return new_payment
        except SQLAlchemyError as e:
            raise DatabaseErrorMapper().map_error(e)

    async def get_payment(self, db):
        pass
        # return await super().get_payment(db)
