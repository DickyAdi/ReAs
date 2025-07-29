import pandas as pd

from domain.io.interfaces.csv import CsvInterface

class CsvLogic:
    def __init__(self, validator:CsvInterface):
        self.validator = validator
    def to_dataframe(self, contents) -> pd.DataFrame:
        decoded = self.validator.decode(contents)
        sanitized = self.validator.sanitize(decoded)
        df = self.validator.to_dataframe(sanitized)
        return df