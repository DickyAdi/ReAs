from typing import Optional, TYPE_CHECKING
from datetime import datetime
from uuid import UUID
from dataclasses import dataclass

if TYPE_CHECKING:
    from domain.entities.transactions import TransactionEntity
    from domain.entities.user import UserEntities
    from domain.entities.subscriptions import SubscriptionPlansEntity


from domain.enums.subscriptions import SubscriptionStatus
from domain.exceptions import UnpaidTransactions

@dataclass
class SubscriptionEntity:
    start_at:datetime
    status: SubscriptionStatus
    end_at:Optional[datetime] = None #default to None, assuming user starts from base tier
    id:Optional[UUID] = None #derived from database
    user_id:Optional[int] = None #derived from database
    transactions_id:Optional[UUID] = None #derived from database
    plans_id:Optional[int] = None #derived from database
    created_at:Optional[datetime] = None #derived from database

    #related entities
    user:Optional["UserEntities"] = None
    plan:Optional["SubscriptionPlansEntity"] = None
    transaction:Optional["TransactionEntity"] = None


    @classmethod
    def create(cls, start_date:datetime, end_date:datetime, status:SubscriptionStatus, plan_id:int, transaction_id:UUID, user_id:int) -> "SubscriptionEntity":
        return cls(
            start_at=start_date,
            end_at=end_date,
            transactions_id=transaction_id,
            plans_id=plan_id,
            user_id=user_id,
            status=status,
        )
    