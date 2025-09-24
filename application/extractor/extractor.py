from typing import Optional, Any

from domain.extractor import ExtractorInterface
from domain.enums.texts import TextSentiment


class ExtractorApplication:
    def __init__(self, service: ExtractorInterface):
        self.service = service

    def extract(
        self,
        reviews: list[Any],
        sentiment: TextSentiment,
        top_n: int,
        language: Optional[str] = "indonesian",
    ):
        review_data = [
            (review.id, review.text)
            for review in reviews
            if review.sentiment == sentiment
        ]
        extracted = self.service.extract(
            review_data=review_data, top_n=top_n, language=language
        )
        return extracted

    def validate_inputs(self, text, sentiment):
        """Validate text and predicted sentiment

        Args:
            text (_type_): Text/review to be extracted.
            sentiment (_type_): Predicted sentiment.
        """
        self.service.validate_inputs(text, predicted_sentiment=sentiment)
