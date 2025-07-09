from fastapi import FastAPI, File, UploadFile, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import gc

from inference import predict
from services.startup import load_model
from services import validation
from services.middleware import LoggingMiddleware
from services.utils import mask_ip
from loggers.log import get_loggers
from config.settings import settings

load_dotenv()

rate_limit_logger = get_loggers('reas.rate_limit')
app_logger = get_loggers('reas.app')

@asynccontextmanager
async def lifespan(app: FastAPI):
    app_logger.info('Starting app...')
    app.state.model = load_model()
    yield
    app_logger.info('Shutting down app...')
    if hasattr(app.state, 'model'):
        app_logger.info('Flushing loaded model...')
        del app.state.model


app = FastAPI(lifespan=lifespan)
rate_limit = Limiter(key_func=get_remote_address)
app.state.limiter = rate_limit

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request:Request, exc:RateLimitExceeded):
    client_ip = mask_ip(get_remote_address(request))
    rate_limit_logger.warning('Rate limit warning for IP: %s, Path: %s', client_ip, request.url.path)
    return JSONResponse(status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        content={
                            'status' : 'error',
                            'code' : 429,
                            'message' : 'Too many requests, chill out please...'
                        })

app.add_middleware(LoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=False,
    allow_methods=['POST'],
    allow_headers=['*'],
)

@app.post("/extract")
@rate_limit.limit("2/second;10/minute;30/day")
async def extract(request: Request, text_column: str, file: UploadFile = File(...)):
    valid = await validation.validate_file(file, text_column)
    if not valid.get('valid'):
        return valid.get('response')
    df = valid.get('data')
    model = request.app.state.model
    predictor = predict.batch_predictor(model, text_column)
    predicted_df, len_valid_mask = predictor.fit_transform(df)
    extractor = predict.topic_extractor(text_column)
    extractor.fit(predicted_df)
    positive_topics = extractor.transform('Positive')
    negative_topics = extractor.transform('Negative')

    response_content = JSONResponse(status_code=status.HTTP_200_OK,
                        content={
                            'status' : 'success',
                            'code' : 200,
                            'message' : 'Extraction successful',
                            'data' : {
                                'positive' : {
                                    'topics' : positive_topics.to_dict(orient='records'),
                                    'count' : int(len(positive_topics))
                                },
                                'negative' : {
                                    'topics' : negative_topics.to_dict(orient='records'),
                                    'count' : int(len(negative_topics))
                                },
                                'number_valid_rows' : int(len_valid_mask)
                            }
                        })
    
    del df, model, predictor, predicted_df, extractor, positive_topics, negative_topics
    gc.collect()
    
    return response_content