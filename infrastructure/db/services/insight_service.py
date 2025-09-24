from sqlalchemy.dialects.postgresql import insert
from sqlalchemy import select, func
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError
from typing import Optional
from datetime import datetime, timezone
from uuid import UUID

from domain.entities.insights import InsightInterface, InsightEntity
from infrastructure.db.models.insight_model import Insights
from infrastructure.db.models.insight_reviews_model import InsightReviews
from infrastructure.db.error_mapper import DatabaseErrorMapper


class InsightService(InsightInterface):
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_insight(
        self,
        obj: InsightEntity,
        commit: Optional[bool] = True,
        generate_defaults: Optional[bool] = False,
    ):
        orm_obj = Insights.from_entity(insight=obj, generate_defaults=generate_defaults)
        if commit:
            try:
                self.db.add(orm_obj)
            except SQLAlchemyError as e:
                exc = DatabaseErrorMapper.map_error(e)
                raise exc
        return orm_obj

    async def upsert_insights(self, insights: list[Insights]):
        list_insights_dict = [insight.to_dict() for insight in insights]
        stmt = insert(Insights).values(list_insights_dict)
        stmt = stmt.on_conflict_do_update(
            index_elements=["dataset_id", "topic"],
            set_={
                "emerging_score": stmt.excluded["emerging_score"],
                "trend_score": stmt.excluded["trend_score"],
                "updated_at": datetime.now(timezone.utc),
            },
        )
        try:
            await self.db.execute(stmt)
        except SQLAlchemyError as e:
            exc = DatabaseErrorMapper.map_error(e)
            raise exc
        return True

    async def upsert_assoc_table(self, assoc_values: list[dict]):
        assoc_val = [
            {"insight_id": assoc.get("insight_id"), "reviews_id": review_id}
            for assoc in assoc_values
            for review_id in assoc["review_id"]
        ]
        stmt = insert(InsightReviews).values(assoc_val)
        stmt = stmt.on_conflict_do_nothing()

        try:
            await self.db.execute(stmt)
        except SQLAlchemyError as e:
            exc = DatabaseErrorMapper.map_error(e)
            raise exc
        return True

    async def get_insights_by_dataset_id(
        self, dataset_id: UUID, offset: Optional[int] = 0, limit: Optional[int] = 0
    ):
        stmt = select(Insights).where(Insights.dataset_id == dataset_id)
        stats_stmt = (
            select(func.count())
            .select_from(Insights)
            .where(Insights.dataset_id == dataset_id)
        )
        if offset:
            stmt = stmt.offset(offset)
        if limit:
            stmt = stmt.limit(limit)

        try:
            res = await self.db.execute(stmt)
            results = res.scalars().all()
            if not offset and not limit:
                return results
            res_stats = await self.db.execute(stats_stmt)
            stats_results = res_stats.scalar_one()
            return {
                "total": stats_results,
                "offset": offset,
                "limit": limit,
                "data": results,
            }
        except SQLAlchemyError as e:
            exc = DatabaseErrorMapper.map_error(e)
            raise exc

    async def get_review_insight(
        self, dataset_id: UUID, offset: Optional[int] = 0, limit: Optional[int] = 0
    ):
        stmt = (
            select(Insights)
            .filter(Insights.dataset_id == dataset_id)
            .options(selectinload(Insights.reviews))
        )
        stats_stmt = (
            select(func.count())
            .select_from(Insights)
            .where(Insights.dataset_id == dataset_id)
        )
        if offset:
            stmt = stmt.offset(offset)
        if limit:
            stmt = stmt.limit(limit)
        stats_res = None
        stats_results = None
        try:
            res = await self.db.execute(stmt)
            data = res.scalars().all()
            if offset or limit:
                stats_res = await self.db.execute(stats_stmt)
                stats_results = stats_res.scalar_one()
        except SQLAlchemyError as e:
            exc = DatabaseErrorMapper.map_error(e)
            raise exc

        data_dict = [
            {
                "insight_id": dt.id,
                "topic": dt.topic,
                "emerging_score": dt.emerging_score,
                "trend_score": dt.trend_score,
                "reviews": [
                    {
                        "review_text": r.text,
                        "review_sentiment": r.sentiment,
                        "review_source": r.provider_ref,
                        "review_timestamp": r.created_at,
                        "review_postdate": r.post_date,
                        "review_platform": r.platform,
                        "review_language": r.language,
                    }
                    for r in dt.reviews
                ],
            }
            for dt in data
        ]
        if offset or limit:
            return {
                "total": stats_results,
                "offset": offset,
                "limit": limit,
                "data": data_dict,
            }
        else:
            return data_dict
