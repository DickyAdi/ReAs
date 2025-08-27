from domain.predictor import PredictorInterface
from domain.ml.interfaces.interface import inferenceInterface


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
