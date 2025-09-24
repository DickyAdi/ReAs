from abc import ABC, abstractmethod

from domain.ml.interfaces.interface import inferenceInterface


class PredictorInterface(ABC):
    """Domain contract for predictor to predict texts sentiment."""

    @abstractmethod
    def clean(self, data: list[dict]) -> list[dict]:
        """Implementation of texts pre-processing/cleaning.

        Args:
            texts (List[str]): Lists of review/text.

        Returns:
            list[str]: Cleaned texts
        """
        pass

    # @abstractmethod
    # def count_valid_mask(self, mask: list[str]) -> int:
    #     """Implementation of counting valid texts/reviews.

    #     Args:
    #         mask (list[str]): List of review/text

    #     Returns:
    #         int: Count of the valid text/review.
    #     """
    #     pass

    @abstractmethod
    def predict(self, data: list[str], model: inferenceInterface) -> list[str]:
        """Implementation of sentiment prediction with injected model.

        Args:
            data (List[str]): List of text/review.
            model (inferenceInterface): Sentiment prediction model.

        Returns:
            List[str]: List of predicted sentiment.
        """
        pass
