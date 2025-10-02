from abc import ABC, abstractmethod
from typing import Optional

from domain.enums.texts import TextSentiment


class ExtractorInterface(ABC):
    """Domain contract for extractor to extract topic from reviews."""

    def validate_inputs(
        self, texts: list[str], predicted_sentiment: list[TextSentiment]
    ):
        if len(texts) != len(predicted_sentiment):
            raise ValueError(
                f"Length of texts doesnt match the length of predicted sentiment. Got, {len(texts)} and {len(predicted_sentiment)}"
            )

    @abstractmethod
    def extract(
        self,
        review_data: list[tuple],
        top_n: int,
        language: Optional[str] = "indonesian",
        coverage_weight: float = 0.6,
        intensity_weight: float = 0.4,
    ):
        pass
