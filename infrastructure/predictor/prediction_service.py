from domain.predictor import PredictorInterface
from domain.ml.interfaces.interface import inferenceInterface


class PredictionService(PredictorInterface):
    def clean(self, data: list[dict]) -> list[dict]:
        cleaned_text = []
        for text in data:
            if (
                isinstance(text.get("text"), str)
                and text.get("text").strip() != ""
                and len(text.get("text").split()) > 1
            ):
                cleaned_text.append(text)
        return cleaned_text

    # def count_valid_mask(self, mask:list[str]) -> int:
    #     """Count number of texts

    #     Args:
    #         mask (list[str]): _description_

    #     Returns:
    #         int: _description_
    #     """
    #     return len(mask)
    def predict(self, data: list[str], model: inferenceInterface) -> list[dict]:
        """Do sentiment inference for the given `data`.

        Args:
            data (list[str]): List of text/review that want to be predicted.
            model (inferenceInterface): Sentiment analysis model.

        Raises:
            ValueError: If `data` is None.

        Returns:
            list[str]: Prediction results.
        """
        if not data or (not isinstance(data, list) and not isinstance(data[0], str)):
            raise ValueError(f"Expected `list[str]`. Got {type(data)}.")
        results = model.predict(data)
        return results
