#Building stage
FROM python:3.12.11-slim AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m nltk.downloader punkt_tab stopwords

COPY . .

#Runtime stage
FROM python:3.12.11-slim AS final

WORKDIR /app
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /root/nltk_data /root/nltk_data
COPY --from=builder /app .

ENV OMP_NUM_THREADS=1 \ 
    OPENBLAS_NUM_THREADS=1 \ 
    MKL_NUM_THREADS=1 \ 
    NUMEXPR_NUM_THREADS=1

ENV PYTHONIOENCODING=UTF-8

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]