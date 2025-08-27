import pandas as pd
import io

from domain.io.interfaces.csv import CsvInterface
from domain.exceptions import InvalidEncodingError, ColumnNotFoundError


class CsvParser(CsvInterface):
    """Infrastructure layer CSV parser service.

    Args:
        CsvInterface (Class): Domain contract.
    """

    def sanitize(
        self, content: io.StringIO
    ) -> io.StringIO:  # skip sanitizing as it an overkill, implement this later when needed or found efficient method
        return content

    def to_dataframe(self, content: io.StringIO) -> pd.DataFrame:
        """Convert StringIO data to Pandas DataFrame

        Args:
            content (io.StringIO): Text or value that want to be converted into Pandas DataFrame

        Returns:
            pd.DataFrame: Converted text or value.
        """
        df = pd.read_csv(content)
        return df

    def decode(self, content: bytes) -> io.StringIO:
        """Decode text bytes to StringIO.

        Args:
            contents (bytes): Text or value that want to be decoded.

        Raises:
            InvalidEncodingError: If passed text/value is not in `utf-8` format.

        Returns:
            io.StringIO: Converted text or value.
        """
        try:
            decoded = content.decode("utf-8")
            return io.StringIO(decoded)
        except UnicodeDecodeError:
            raise InvalidEncodingError

    def get_text(self, df: pd.DataFrame, text_column: str) -> list[str]:
        """Get list of text on the given dataframe `df`.

        Args:
            df (pd.DataFrame): DataFrame of the users csv.
            text_column (str): The column name that contains the text/review.

        Raises:
            ColumnNotFoundError: If `text_column` is not in `df` DataFrame columns.

        Returns:
            list[str]: List of the review/text.
        """
        if text_column not in df.columns.tolist():
            raise ColumnNotFoundError(
                text_column=text_column, available_column=df.columns.tolist()
            )
        return df[text_column].tolist()
