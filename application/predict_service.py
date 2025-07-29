import pandas as pd

from core.predict import PredictLogic
from domain.ml.interfaces.interface import inferenceInterface

class PredictService:
    def __init__(self, model:inferenceInterface):
        self.model = model
        self.logic = PredictLogic()
    def run(self, df, text_column) -> tuple[pd.DataFrame, int]:
        if not hasattr(self.model, 'predict') and not callable(self.model.predict):
            raise NotImplementedError(f'Model predict interface is not callable.')
        predicted_df, len_valid_mask = self.logic.predict(df, text_column, self.model)
        return predicted_df, len_valid_mask