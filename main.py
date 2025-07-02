from fastapi import FastAPI, File, UploadFile, Request, status
from fastapi.responses import JSONResponse
import io
import pandas as pd

from inference import predict
from utils import get_random_file_name
from services.startup import load_assets

app = FastAPI()

@app.on_event('startup')
def startup():
    load_assets(app)


@app.post("/extract")
async def extract(request: Request, file: UploadFile = File(...)):
    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        print(f"something went wrong {e}")
    model = request.app.state.model
    predictor = predict.batch_predictor(model)
    predicted_df = predictor.fit_transform(df)
    extractor = predict.topic_extractor()
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