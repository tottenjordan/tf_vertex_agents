
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.10

COPY src/prediction /app
COPY requirements.txt /app/requirements.txt

RUN pip3 install -r /app/requirements.txt
