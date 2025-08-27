from abc import ABC, abstractmethod


class CsvInterface(ABC):
    @abstractmethod
    def sanitize(self, content):
        pass

    @abstractmethod
    def to_dataframe(self, content):
        pass

    @abstractmethod
    def decode(self, content):
        pass

    @abstractmethod
    def get_text(self, df, text_column):
        pass
