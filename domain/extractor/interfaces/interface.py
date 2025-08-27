from abc import ABC, abstractmethod
from typing import Any

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

    # @abstractmethod
    # def get_text_by_sentiment(
    #     self,
    #     sentiment: TextSentiment,
    #     texts: list[str],
    #     predicted_sentiment: list[TextSentiment],
    # ) -> list[str]:
    #     """Implementation of getting text by chosen sentiment.

    #     Args:
    #         sentiment (TextSentiment): Chosen sentiment enum.
    #         texts (list[str]): List of review/text.
    #         predicted_sentiment (list[TextSentiment]): List of predicted sentiment from the given `texts` list.

    #     Raises:
    #         ValueError: If `len(texts)` and `len(predicted_sentiment)` doesnt match.

    #     Returns:
    #         list[str]: List of chosen sentiment text.
    #     """
    #     pass

    @abstractmethod
    def extract(self, text: list[str]) -> tuple[Any, Any]:
        """Implementation of extracting topic insights from the given list of text.

        Args:
            text (list[str]): List of reviews that need to be extracted.

        Returns:
            tuple[Any, Any]: A tuple of pandas DataFrame that contains trending topics and frequently mentioned topics.
        """
        pass
