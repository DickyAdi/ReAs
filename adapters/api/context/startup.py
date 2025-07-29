from concurrent.futures import ThreadPoolExecutor

from application.load_model_service import ModelLoader
from infrastructure.ml.inference import inferenceModel
from config.settings import settings

def load_model():
    inference_model = inferenceModel
    model_loader = ModelLoader(inference_model)
    model = model_loader.get_model()
    return model

def get_executor(max_workers=settings.concurrent_worker):
    return ThreadPoolExecutor(max_workers=max_workers)