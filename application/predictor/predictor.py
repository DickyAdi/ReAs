from typing import Any

from domain.predictor import PredictorInterface
from domain.ml.interfaces.interface import inferenceInterface
from domain.entities.reviews import ReviewEntity


class PredictorApplication:
    def __init__(self, service: PredictorInterface):
        self.service = service

    def predict(self, texts: list[str], model: inferenceInterface) -> list[dict]:
        """Run sentiment prediction for the given `texts` with `model`.

        Args:
            texts (list[str]): List of review/text to be predicted.
            model (inferenceInterface): Model for sentiment prediction.

        Returns:
            list[str]: List of predicted sentiment.
        """
        results = self.service.predict(data=texts, model=model)
        return results

    def predict_from_class(
        self, reviews: list[ReviewEntity], model: inferenceInterface
    ):
        texts = [getattr(review, "text") for review in reviews]
        results = self.service.predict(data=texts, model=model)
        return results

    def clean(self, reviews: list[dict]):
        # return self.service.clean(data=reviews)
        return self.service.clean(data=reviews)
