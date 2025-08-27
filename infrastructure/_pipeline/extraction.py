import pandas as pd

from application.pipeline.extraction import (
    ExtractionPipelineFlow,
    ExtractionPipelineDictType,
)
from app.application._predict_service import PredictService
from app.application._extract_service import ExtractService
from infrastructure.ml.inference import inferenceModel
from infrastructure.topics.interface.extract import ExtractTopics


class ExtractionPipeline:
    def extract(self, df: pd.DataFrame, text_column: str) -> ExtractionPipelineDictType:
        predictor = PredictService(model=inferenceModel())
        extractor = ExtractService(
            extractor=ExtractTopics(df=df, text_column=text_column)
        )
        pipe = ExtractionPipelineFlow(predictor=predictor, extractor=extractor)
        result = pipe.run(df, text_column)
        return result
