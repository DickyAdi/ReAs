from abc import ABC, abstractmethod

class inferenceInterface(ABC):
    @abstractmethod
    def predict(self, texts:str | list[str]) -> str | list[str]:
        pass
    