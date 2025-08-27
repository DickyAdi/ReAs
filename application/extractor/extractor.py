from typing import Any

from domain.extractor import ExtractorInterface


class ExtractorApplication:
    def __init__(self, service: ExtractorInterface):
        self.service = service

    def extract(self, text: list[str]) -> tuple[Any, Any]:
        """Extract insights from the given `text`

        Args:
            text (list[str]): List of review/text.

        Returns:
            tuple[Any, Any]: Tuple of trend and frequent Pandas DataFrame.
        """
        trend_df, frequent_df = self.service.extract(text=text)
        return (trend_df, frequent_df)

    # def get_text_by_sentiment(self, sentiment:str):

    def validate_inputs(self, text, sentiment):
        """Validate text and predicted sentiment

        Args:
            text (_type_): Text/review to be extracted.
            sentiment (_type_): Predicted sentiment.
        """
        self.service.validate_inputs(text, predicted_sentiment=sentiment)
