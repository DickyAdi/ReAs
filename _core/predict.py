import pandas as pd

from domain.ml.interfaces.interface import inferenceInterface
from typing import Tuple


class PredictLogic:
    def __init__(self):
        self.len_valid_mask = 0

    def clean(self, texts: pd.Series) -> Tuple[pd.Series, pd.Series]:
        text_series = texts.str.lower()
        empty_mask = text_series.apply(
            lambda x: isinstance(x, str) and x.strip() != "" and len(x.split()) > 1
        )
        return text_series, empty_mask

    def count_valid_mask(self, mask: pd.Series):
        self.len_valid_mask = mask[mask == True].count()

    def predict(
        self, df: pd.DataFrame, text_column: str, model: inferenceInterface
    ) -> Tuple[pd.DataFrame, int]:
        if df is None or not isinstance(df, pd.DataFrame):
            raise ValueError(
                f"Value error: `df` is empty or not pandas DataFrame. Got {type(df)}"
            )
        if not text_column:
            raise ValueError("Value error: `text_column` is empty")
        if text_column not in df.columns.tolist():
            raise KeyError(f"{text_column} not in DataFrame columns.")
        text_series, valid_mask = self.clean(df[text_column])
        self.count_valid_mask(valid_mask)
        df.loc[valid_mask, "prediction"] = model.predict(
            text_series[valid_mask].tolist()
        )
        df.loc[~valid_mask, "prediction"] = None
        return df, self.len_valid_mask
