from typing import Protocol, Self, Any, Union
from domain.entities.data_source_logs import DataSourceInterface
from domain.entities.datasets import DatasetInterface
from domain.entities.insights import InsightInterface

# from domain.entities.payments import PaymentInterface
from domain.entities.reviews import ReviewInterface
# from domain.entities.subscriptions import (
#     SubscriptionPlansInterface,
#     SubscriptionInterface,
# )
# from domain.entities.tiers import TiersInterface
# from domain.entities.transactions import TransactionInterface
# from domain.entities.user import UserInterface

# ? Partial refactor, the UoW only for the newest update which are data source, dataset, insight, and review.


class UnitOfWorkInterface(Protocol):
    data_sources: DataSourceInterface
    datasets: DatasetInterface
    insights: InsightInterface
    # payments: PaymentInterface
    reviews: ReviewInterface
    # subscriptions: SubscriptionInterface
    # plans: SubscriptionPlansInterface
    # tiers: TiersInterface
    # transactions: TransactionInterface
    # users: UserInterface

    async def __aenter__(self) -> Self: ...
    async def __aexit__(self, exc_type, exc_val, exc_trace) -> None: ...
    async def commit(self) -> None: ...
    async def rollback(self) -> None: ...
    async def flush(self) -> None: ...
    async def add(self, obj: Union[Any, list[Any]]): ...
