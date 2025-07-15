from dotenv import load_dotenv
load_dotenv()


from fastapi import FastAPI, File, UploadFile, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from concurrent.futures import ThreadPoolExecutor
import asyncio
import gc

from inference.pipeline import extraction_pipeline
from services.startup import load_model, log_thread_usage
from services import validation
from services.middleware import LoggingMiddleware
from services.utils import mask_ip
from loggers.log import get_loggers
from config.settings import settings


rate_limit_logger = get_loggers('reas.rate_limit')
app_logger = get_loggers('reas.app')

@asynccontextmanager
async def lifespan(app: FastAPI):
    app_logger.info('Starting app...')
    app_logger.info('Configured  with %d concurrent worker.', settings.concurrent_worker)
    app.state.model = load_model()
    log_thread_usage()
    app.state.executor = ThreadPoolExecutor(max_workers=settings.concurrent_worker)
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
    try:
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(request.app.state.executor, extraction_pipeline, model, df, text_column)
    except Exception as e:
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            content={
                                'status' : 'error',
                                'code' : 500,
                                'message' : f'Internal server error. {str(e)}'
                            })

    response_content = JSONResponse(status_code=status.HTTP_200_OK,
                        content={
                            'status' : 'success',
                            'code' : 200,
                            'message' : 'Extraction successful',
                            'data' : {
                                'positive' : {
                                    'trend_topics' : results['positive_trend_topics'].to_dict(orient='records'),
                                    'frequent_topics' : results['positive_frequent_topics'].to_dict(orient='records'),
                                    'count' : results['n_positive']
                                },
                                'negative' : {
                                    'trend_topics' : results['negative_trend_topics'].to_dict(orient='records'),
                                    'frequent_topics' : results['negative_frequent_topics'].to_dict(orient='records'),
                                    'count' : results['n_negative']
                                },
                                'number_valid_rows' : int(results['len_valid_mask'])
                            }
                        })
    
    del df, model, results
    gc.collect()
    
    return response_content