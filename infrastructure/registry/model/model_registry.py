from infrastructure.ml.inference import inferenceModel

from domain.registry.model import ModelRegistryInterface


class ModelRegistry(ModelRegistryInterface):
    def __init__(self):
        self._registry = {"default": inferenceModel()}

    @property
    def model_registry(self):
        return self._registry
