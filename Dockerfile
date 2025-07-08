FROM python:3.12.11-slim

WORKDIR /app

#Initiate empty log dir
RUN mkdir -p logs

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]