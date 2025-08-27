from application.use_cases.pipeline import PipelineExtractCsvFlow

from infrastructure.extractor import ExtractionService
from infrastructure.io.parse_csv import CsvParser
from infrastructure.predictor import PredictionService


def get_extract_csv_flow():
    return PipelineExtractCsvFlow(
        csv_service=CsvParser(),
        prediction_service=PredictionService(),
        extraction_service=ExtractionService(),
    )
