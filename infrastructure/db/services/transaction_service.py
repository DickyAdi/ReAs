from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Optional

from ..models.transaction_model import Transactions

from domain.entities.transactions.interfaces import TransactionInterface
from domain.entities.transactions import TransactionEntity


class TransactionService(TransactionInterface):
    async def get_transaction_by_idempotency_key(
        self, db: AsyncSession, idempotency_key: str
    ) -> Transactions | None:
        """Get transaction by idempotency key.

        Args:
            db (AsyncSession): Async session of the database.
            idempotency_key (str): User unique idempotency key.

        Returns:
            Transactions | None: If match, return Transaction. Else, return None.
        """
        stmnt = select(Transactions).where(
            Transactions.idempotency_key == idempotency_key
        )
        res = await db.execute(stmnt)
        trx = res.scalar_one_or_none()
        return trx

    async def create_idempotent_transaction(
        self,
        db: AsyncSession,
        transaction: TransactionEntity,
        commit: Optional[bool] = True,
    ) -> Transactions:
        """Create idempotent transaction. If a same transaction present via idempotency key checking, return the matching transaction.
        Else, create new transaction.

        Args:
            db (AsyncSession): Async session of the database
            transaction (TransactionEntity): Transaction entity object.
            commit (Optional[bool], optional): Whether to directly commit or flush only. Defaults to True.

        Returns:
            Transactions: Created idempotent transaction.
        """
        idempotent_trx = await self.get_transaction_by_idempotency_key(
            db=db, idempotency_key=transaction.idempotency_key
        )
        if idempotent_trx:
            return idempotent_trx
        new_trx = Transactions.from_entity(transaction=transaction)
        db.add(new_trx)
        await db.flush()
        if commit:
            await db.commit()
            await db.refresh(new_trx)
        return new_trx

    async def get_transaction(self, db):
        pass
        # return await super().get_transaction(db)
