from typing import Optional, Union
from uuid import UUID

from domain.uow import UnitOfWorkInterface
from domain.entities.insights import InsightEntity

from application.rules import enforce_transaction


class InsightApplication:
    def __init__(self, uow: UnitOfWorkInterface):
        self.uow = uow

    @enforce_transaction
    async def create_insight(
        self,
        topic: str,
        emerging_score: float,
        trend_score: float,
        dataset_id: UUID,
        commit: Optional[bool] = True,
        generate_defaults: Optional[bool] = False,
    ):
        domain_obj = InsightEntity.create(
            topic=topic,
            emerging_score=emerging_score,
            trend_score=trend_score,
            dataset_id=dataset_id,
        )
        orm_obj = await self.uow.insights.create_insight(
            domain_obj, commit=commit, generate_defaults=generate_defaults
        )
        if commit:
            await self.uow.commit()
        return orm_obj

    async def create_insight_from_dict(
        self,
        insight_data: Union[dict, list[dict]],
        dataset_id: UUID,
        commit: Optional[bool] = True,
        generate_defaults: Optional[bool] = False,
    ):
        if isinstance(insight_data, list):
            insights = [
                await self.create_insight(
                    topic=data.get("topic"),
                    emerging_score=data.get("emerging_score"),
                    trend_score=data.get("trend_score"),
                    dataset_id=dataset_id,  # injected
                    commit=commit,
                    generate_defaults=generate_defaults,
                )
                for data in insight_data
            ]
        else:
            insights = [
                await self.create_insight(
                    topic=insight_data.get("topic"),
                    emerging_score=insight_data.get("emerging_score"),
                    trend_score=insight_data.get("trend_score"),
                    dataset_id=dataset_id,  # injected
                    commit=commit,
                    generate_defaults=generate_defaults,
                )
            ]
        return insights

    @enforce_transaction
    async def upsert_insights(
        self,
        insights_data: list[dict],
        insight_review_mappings: list[list[UUID]],
        dataset_id: UUID,
    ):
        if len(insights_data) != len(insight_review_mappings):
            raise ValueError(
                "given `insights_data` and `insight_review_mappings` length not match"
            )
        insights = await self.create_insight_from_dict(
            insight_data=insights_data,
            commit=False,
            generate_defaults=True,
            dataset_id=dataset_id,
        )
        upserted_id = await self.uow.insights.upsert_insights(insights=insights)
        assoc_val = [
            {"insight_id": insight_id, "review_id": reviews_id}
            for insight_id, reviews_id in zip(upserted_id, insight_review_mappings)
        ]
        _is_assoc_upserted = await self.uow.insights.upsert_assoc_table(
            assoc_values=assoc_val
        )
        return (upserted_id is not None) and _is_assoc_upserted

    @enforce_transaction
    async def get_insights_by_dataset_id(
        self,
        dataset_id: UUID,
        offset: Optional[int] = 0,
        limit: Optional[int] = 0,
        to_pub: Optional[bool] = False,
    ):
        insight_data = await self.uow.insights.get_insights_by_dataset_id(
            dataset_id=dataset_id, offset=offset, limit=limit
        )
        if to_pub:
            if isinstance(insight_data, dict):
                insight_data["data"] = [data.to_dict() for data in insight_data["data"]]
            else:
                insight_data = [data.to_dict() for data in insight_data]
            return insight_data
        else:
            return insight_data

    @enforce_transaction
    async def get_review_insight(
        self, dataset_id: UUID, offset: Optional[int] = 0, limit: Optional[int] = 0
    ):
        return await self.uow.insights.get_review_insight(
            dataset_id=dataset_id, offset=offset, limit=limit
        )
