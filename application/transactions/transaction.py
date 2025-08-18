from decimal import Decimal
from typing import Optional
from uuid import UUID

from domain.entities.transactions.interfaces import TransactionInterface
from domain.entities.transactions import TransactionEntity
from domain.enums.transactions import TransactionStatus
from domain.entities.transactions import PlanSnapshotDict


class TransactionApplication:
    """Transaction application for transaction services."""

    def __init__(self, transaction_service: TransactionInterface):
        self.transaction_service = transaction_service

    async def issue_transaction(
        self,
        db,
        amount: Decimal,
        idempotency_key: UUID,
        plan_snapshot: PlanSnapshotDict,
        payment_id: UUID,
        trx_status: Optional[TransactionStatus] = TransactionStatus.pending,
    ):
        """Issue/create new transaction.

        Args:
            db (Any): Async session of the database.
            amount (Decimal): Amount that was/has been charged to the user.
            idempotency_key (UUID): Random unique UUID to make the transaction idempotent.
            plan_snapshot (PlanSnapshotDict): Latest and used subscription plan for the transaction.
            payment_id (UUID): Created payment intent for the user.
            trx_status (Optional[TransactionStatus], optional): Transactions status. Defaults to TransactionStatus.pending.

        Returns:
            Transaction: Infrastructure layer Transaction ORM instance.
        """
        trx_obj = TransactionEntity.create(
            status=trx_status,
            amount=amount,
            payment_id=payment_id,
            plan_snapshot=plan_snapshot,
            idempotency_key=idempotency_key,
        )
        trx = await self.transaction_service.create_idempotent_transaction(
            db=db, transaction=trx_obj, commit=False
        )
        return trx
