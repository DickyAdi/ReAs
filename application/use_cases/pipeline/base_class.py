from typing import Any, TypedDict


from application.predictor import PredictorApplication
from application.extractor import ExtractorApplication

from domain.predictor import PredictorInterface
from domain.extractor import ExtractorInterface
from domain.ml.interfaces.interface import inferenceInterface
from domain.enums.texts import TextSentiment


class ExtractionPipelineDictType(TypedDict):
    n_positive: int
    n_negative: int
    positive_trend_topics: Any
    positive_frequent_topics: Any
    negative_trend_topics: Any
    negative_frequent_topics: Any
    total_valid_reviews: int


class BasePipelineFlow:
    def __init__(
        self,
        prediction_service: PredictorInterface,
        extraction_service: ExtractorInterface,
    ):
        self.prediction_app = PredictorApplication(service=prediction_service)
        self.extraction_app = ExtractorApplication(service=extraction_service)
        # self.model = model_service

    def run_predict_extract(
        self, text: list[str], model: inferenceInterface
    ) -> ExtractionPipelineDictType:
        """Run prediction and extract topic insights for both Positive and Negative sentiment.

        Args:
            text (list[str]): List of review/text.
            model (inferenceInterface): Chosen model for sentiment prediction.

        Returns:
            ExtractionPipelineDictType: Result dictionary for both Positive and Negative sentiment.
        """
        predicted = self.prediction_app.predict(texts=text, model=model)
        # positive_texts = [
        #     sentiment.get("text")
        #     for sentiment in predicted
        #     if sentiment.get("sentiment") == "Positive"
        # ]
        # negative_texts = [
        #     sentiment.get("text")
        #     for sentiment in predicted
        #     if sentiment.get("sentiment") == "Negative"
        # ]
        positive_texts = [
            sentiment.get("text")
            for sentiment in predicted
            if sentiment.get("sentiment") == TextSentiment.positive
        ]
        negative_texts = [
            sentiment.get("text")
            for sentiment in predicted
            if sentiment.get("sentiment") == TextSentiment.negative
        ]
        positive_trend, positive_frequent = self.extraction_app.extract(positive_texts)
        negative_trend, negative_frequent = self.extraction_app.extract(negative_texts)
        return {
            "n_positive": len(positive_texts),
            "n_negative": len(negative_texts),
            "positive_trend_topics": positive_trend,
            "positive_frequent_topics": positive_frequent,
            "negative_trend_topics": negative_trend,
            "negative_frequent_topics": negative_frequent,
            "total_valid_reviews": len(predicted),
        }
