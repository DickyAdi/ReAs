from domain.io.interfaces.csv import CsvInterface
from core.csv import CsvLogic

class ParseCsvService:
    def __init__(self, content, validator:CsvInterface):
        self.content = content
        self.logic = CsvLogic(validator)
    def run(self):
        df = self.logic.to_dataframe(self.content)
        return df