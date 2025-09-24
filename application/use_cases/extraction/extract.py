from typing import Any, Optional
from uuid import UUID

from domain.extractor import ExtractorInterface
from domain.uow import UnitOfWorkInterface
from domain.enums.texts import TextSentiment

from config.settings import settings

from application.reviews import ReviewApplication
from application.extractor import ExtractorApplication


class ExtractionUseCase:
    def __init__(
        self,
        extraction_service: ExtractorInterface,
    ):
        self.extraction_app = ExtractorApplication(service=extraction_service)

    def extract(
        self,
        reviews: list[Any],
        top_n: int = settings.n_topic_store,
        language: Optional[str] = "indonesian",
    ):
        positive_extract = self.extraction_app.extract(
            reviews=reviews,
            sentiment=TextSentiment.positive,
            top_n=top_n,
            language=language,
        )
        negative_extract = self.extraction_app.extract(
            reviews=reviews,
            sentiment=TextSentiment.negative,
            top_n=top_n,
            language=language,
        )
        return (positive_extract, negative_extract)

    async def get_reviews_from_dataset(
        self,
        uow: UnitOfWorkInterface,
        dataset_id: UUID,
    ):
        review_app = ReviewApplication(uow=uow)
        # return await review_app.get_reviews_by_dataset_id(
        #     dataset_id=dataset_id, offset=offset, limit=limit
        # )
        return await review_app.fetch_reviews_by_dataset_id(dataset_id=dataset_id)
