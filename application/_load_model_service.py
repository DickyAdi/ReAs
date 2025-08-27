from domain.ml.interfaces.interface import inferenceInterface

class ModelLoader:
    def __init__(self, model:inferenceInterface):
        self.model = model
    def get_model(self) -> inferenceInterface:
        return self.model