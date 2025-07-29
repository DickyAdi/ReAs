import pandas as pd
import io

from domain.io.interfaces.csv import CsvInterface

class CsvParser(CsvInterface):
    def sanitize(self, contents:io.StringIO) -> io.StringIO: #skip sanitizing as it an overkill, implement this later when needed or found efficient method
        return contents
    
    def to_dataframe(self, content:io.StringIO) -> pd.DataFrame:
        df = pd.read_csv(content)
        return df

    def decode(self, contents:bytes) -> io.StringIO:
        try:
            decoded = contents.decode('utf-8')
            return io.StringIO(decoded)
        except UnicodeDecodeError:
            raise ValueError('Decoding failed, expected utf-8')
        