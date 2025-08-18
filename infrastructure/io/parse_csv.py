import pandas as pd
import io

from domain.io.interfaces.csv import CsvInterface


class CsvParser(CsvInterface):
    """Infrastructure layer CSV parser service.

    Args:
        CsvInterface (Class): Domain contract.
    """

    def sanitize(
        self, contents: io.StringIO
    ) -> io.StringIO:  # skip sanitizing as it an overkill, implement this later when needed or found efficient method
        return contents

    def to_dataframe(self, content: io.StringIO) -> pd.DataFrame:
        """Convert StringIO data to Pandas DataFrame

        Args:
            content (io.StringIO): Text or value that want to be converted into Pandas DataFrame

        Returns:
            pd.DataFrame: Converted text or value.
        """
        df = pd.read_csv(content)
        return df

    def decode(self, contents: bytes) -> io.StringIO:
        """Decode text bytes to StringIO.

        Args:
            contents (bytes): Text or value that want to be decoded.

        Raises:
            ValueError: If passed text/value is not in `utf-8` format.

        Returns:
            io.StringIO: Converted text or value.
        """
        try:
            decoded = contents.decode("utf-8")
            return io.StringIO(decoded)
        except UnicodeDecodeError:
            raise ValueError("Decoding failed, expected utf-8")
