from sqlalchemy import select, func, insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional
from uuid import UUID

from domain.entities.reviews import ReviewInterface, ReviewEntity
from domain.exceptions import PaginationTypeError
from infrastructure.db.models import Reviews
from infrastructure.db.error_mapper import DatabaseErrorMapper


class ReviewService(ReviewInterface):
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_review(
        self,
        review: ReviewEntity,
        commit: Optional[bool] = True,
        generate_defaults: Optional[bool] = False,
    ) -> "Reviews":
        """Create ORM Datasets instance and the created instance will be added to the database session

        Args:
            review (ReviewEntity): Domain review instance.

        Raises:
            DatabaseErrorMapper.map_error: Will raise specific domain exception if encounter any SQLAlchemyError

        Returns:
            Reviews ORM instance
        """
        obj = Reviews.from_entity(review=review, generate_defaults=generate_defaults)
        if commit:
            try:
                # async with self.uow as uow:
                self.db.add(obj)
            except SQLAlchemyError as e:
                exc = DatabaseErrorMapper.map_error(e)
                raise exc

        return obj

    async def batch_insert(self, reviews: list[Reviews]):
        """Batch insert list of Reviews ORM instances

        Args:
            reviews (list[Reviews]): List of ORM instances

        Raises:
            DatabaseErrorMapper.map_error: Will raise specific domain exception if encounter any SQLAlchemyError

        Returns:
            Will return True if success
        """
        data = [review.to_dict() for review in reviews]
        stmt = insert(Reviews).values(data)
        # async with self.uow as uow:
        try:
            await self.db.execute(stmt)
        except SQLAlchemyError as e:
            exc = DatabaseErrorMapper.map_error(e)
            raise exc
        return True

    async def get_reviews_by_dataset_id(
        self,
        dataset_id: UUID,
        offset: Optional[int] = 0,
        limit: Optional[int] = 0,
    ):
        """Get reviews by dataset id with pagination

        Args:
            dataset_id (UUID): Dataset identifiers
            offset (Optional[int], optional): Starts from row..... Defaults to 0.
            limit (Optional[int], optional): Limit number of row to..... Defaults to 0.

        Raises:
            PaginationTypeError: If `offset` or `limit` provided violate the rules.
            exc: Translated database error to domain error

        Returns:
            Will return list of Reviews if not paginated, else will return dictionary containing `total`, `offset`, `limit`, and `data`
        """
        if offset < 0:
            raise PaginationTypeError(
                message="`offset` must be non-negative integer value", offset=offset
            )
        if limit < 1 or limit > 100:
            raise PaginationTypeError(
                message="`limit` must be between 1 and 100", limit=limit
            )
        stmt = select(Reviews).where(Reviews.dataset_id == dataset_id)
        stmt_stats = (
            select(func.count())
            .select_from(Reviews)
            .where(Reviews.dataset_id == dataset_id)
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
            res_stats = await self.db.execute(stmt_stats)
            count_results = res_stats.scalar_one()
            return {
                "total": count_results,
                "offset": offset,
                "limit": limit,
                "data": results,
            }
        except SQLAlchemyError as e:
            exc = DatabaseErrorMapper.map_error(e)
            raise exc

    async def count_review_in_dataset(self, dataset_id: UUID) -> int:
        stmt = (
            select(func.count())
            .select_from(Reviews)
            .where(Reviews.dataset_id == dataset_id)
        )
        res = await self.db.execute(stmt)
        return res.scalar_one()
