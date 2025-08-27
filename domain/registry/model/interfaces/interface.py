from abc import ABC, abstractmethod
from domain.ml.interfaces.interface import inferenceInterface


class ModelRegistryInterface(ABC):
    @property
    @abstractmethod
    def model_registry(self):
        pass

    def get_model(self, model_name: str) -> inferenceInterface:
        """Get model from model registry based on the given model name.

        Args:
            model_name (str): model name that want to be retrieved.

        Raises:
            TypeError: If `model_registry` is not a dictionary.

        Returns:
            Selected model
        """
        if not isinstance(self.model_registry, dict):
            raise TypeError(
                f"{self.__class__.__name__}.model_registry must be a dict, "
                f"but got {type(self.model_registry).__name__}"
            )
        return self.model_registry[model_name]
