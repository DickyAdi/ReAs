from typing import Optional

from domain.predictor import PredictorInterface
from domain.registry.model import ModelRegistryInterface

from application.predictor import PredictorApplication


class PredictionUseCase:
    def __init__(
        self,
        model_registry: ModelRegistryInterface,
        predict_service: PredictorInterface,
    ):
        self.model_registry = model_registry
        self.predict_app = PredictorApplication(service=predict_service)

    def predict(self, texts: list[str], model_name: Optional[str] = "default"):
        # return self.predict_service.predict(data=texts, model=model)
        return self.predict_app.predict(
            texts=texts, model=self.model_registry.get_model(model_name=model_name)
        )

    def clean(self, reviews: list[dict]):
        return self.predict_app.clean(reviews)
