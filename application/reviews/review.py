from typing import Optional, Union, Any
from datetime import datetime
from uuid import UUID

from domain.entities.reviews import ReviewEntity
from domain.enums.texts import TextLanguage, TextPlatform, TextSentiment
from domain.uow import UnitOfWorkInterface
from application.rules import enforce_transaction


class ReviewApplication:
    def __init__(self, uow: UnitOfWorkInterface):
        self.uow = uow

    @enforce_transaction
    async def create_review(
        self,
        text: str,
        post_date: datetime,
        sentiment: TextSentiment,
        platform: TextPlatform,
        language: TextLanguage,
        provider_ref: UUID,
        dataset_id: UUID,
        rating: Optional[int] = None,
        commit: Optional[bool] = True,
        generate_defaults: Optional[bool] = False,
    ) -> Any:
        domain_obj = ReviewEntity.create(
            text=text,
            post_date=post_date,
            platform=platform,
            sentiment=sentiment,
            language=language,
            provider_ref=provider_ref,
            dataset_id=dataset_id,
            rating=rating,
        )
        obj = await self.uow.reviews.create_review(
            review=domain_obj, commit=commit, generate_defaults=generate_defaults
        )
        if commit:
            await self.uow.commit()

        return obj

    @enforce_transaction
    async def create_reviews_from_dict(
        self,
        reviews: Union[list[dict], dict],
        commit: Optional[bool] = True,
        generate_defaults: Optional[bool] = False,
    ) -> list[Any]:
        if isinstance(reviews, list):
            obj = [
                await self.create_review(
                    text=review.get("text"),
                    post_date=review.get("post_date"),
                    platform=review.get("platform"),
                    sentiment=review.get("sentiment"),
                    provider_ref=review.get("provider_ref"),
                    dataset_id=review.get("dataset_id"),
                    rating=review.get("rating"),
                    language=review.get("language"),
                    commit=commit,
                    generate_defaults=generate_defaults,
                )
                for review in reviews
            ]
        else:
            obj = [
                await self.create_review(
                    text=reviews.get("text"),
                    post_date=reviews.get("post_date"),
                    platform=reviews.get("platform"),
                    sentiment=reviews.get("sentiment"),
                    provider_ref=reviews.get("provider_ref"),
                    dataset_id=reviews.get("dataset_id"),
                    rating=reviews.get("rating"),
                    language=reviews.get("language"),
                    commit=commit,
                    generate_defaults=generate_defaults,
                )
            ]
        return obj

    @enforce_transaction
    async def insert_reviews_from_dict(
        self,
        reviews: Union[list[dict], dict],
        get_id: Optional[bool] = False,
        get_dataset_id: Optional[bool] = False,
        generate_defaults: Optional[bool] = False,
    ):
        reviews = await self.create_reviews_from_dict(
            reviews=reviews, commit=False, generate_defaults=generate_defaults
        )
        batch_size = 5000
        if len(reviews) > batch_size:
            for i in range(0, len(reviews), batch_size):
                is_inserted = await self.uow.reviews.batch_insert(
                    reviews[i : i + batch_size]
                )
                if not is_inserted:
                    raise RuntimeError("Reviews not inserted")
            return is_inserted
        else:
            is_inserted = await self.uow.reviews.batch_insert(reviews)
        if get_id:
            reviews_id = [rev.id for rev in reviews]
            return reviews_id
        elif get_dataset_id:
            dataset_id = reviews[0].dataset_id
            return dataset_id
        else:
            return is_inserted

    @enforce_transaction
    async def fetch_reviews_by_dataset_id(self, dataset_id: UUID):
        reviews = []
        offset = 0
        limit = 100
        while True:
            chunk = await self.uow.reviews.get_reviews_by_dataset_id(
                dataset_id=dataset_id, offset=offset, limit=limit
            )
            review = chunk["data"]
            reviews.extend(review)
            if not review:
                break
            offset += limit
        return reviews

    @enforce_transaction
    async def count_review_in_dataset(self, dataset_id: UUID) -> int:
        return await self.uow.reviews.count_review_in_dataset(dataset_id=dataset_id)
