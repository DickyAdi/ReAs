from application.io import CsvApplication

from domain.io.interfaces.csv import CsvInterface
from domain.predictor import PredictorInterface
from domain.extractor import ExtractorInterface
from domain.ml.interfaces.interface import inferenceInterface

from .base_class import BasePipelineFlow, ExtractionPipelineDictType


class PipelineExtractCsvFlow(BasePipelineFlow):
    """Insights extraction pipeline from user uploaded csv."""

    def __init__(
        self,
        csv_service: CsvInterface,
        prediction_service: PredictorInterface,
        extraction_service: ExtractorInterface,
    ):
        super().__init__(
            prediction_service=prediction_service,
            extraction_service=extraction_service,
        )
        self.csv_app = CsvApplication(service=csv_service)

    def __call__(
        self, content: bytes, text_column: str, model: inferenceInterface
    ) -> ExtractionPipelineDictType:
        """_summary_

        Args:
            content (bytes): Users csv in bytes.
            text_column (str): Csv column name that contains the review.
            model (inferenceInterface): Chosen model for sentiment prediction.

        Returns:
            ExtractionPipelineDictType: Extracted insights.
        """
        texts = self.csv_app.get_text(content=content, text_column=text_column)
        result = self.run_predict_extract(text=texts, model=model)
        return result
