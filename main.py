from fastapi import FastAPI, File, UploadFile, Request, status
from fastapi.responses import JSONResponse
import io
import pandas as pd
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from inference import predict
from services.startup import load_model
from services import validation

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model()
    yield

app = FastAPI(lifespan=lifespan)

@app.post("/extract")
async def extract(request: Request, text_column: str, file: UploadFile = File(...)):
    valid = await validation.validate_file(file, text_column)
    if not valid.get('valid'):
        return valid.get('response')
    df = valid.get('data')
    model = request.app.state.model
    predictor = predict.batch_predictor(model, text_column)
    predicted_df = predictor.fit_transform(df)
    extractor = predict.topic_extractor(text_column)
    extractor.fit(predicted_df)
    positive_topics = extractor.transform('Positive')
    negative_topics = extractor.transform('Negative')

    return JSONResponse(status_code=status.HTTP_200_OK,
                        content={
                            'status' : 'success',
                            'code' : 200,
                            'message' : 'Extraction successful',
                            'data' : {
                                'positive_topics' : positive_topics.to_dict(orient='records'),
                                'negative_topics' : negative_topics.to_dict(orient='records')
                            }
                        })