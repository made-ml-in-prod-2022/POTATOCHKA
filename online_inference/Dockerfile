FROM python:3.6-slim-stretch
COPY ml_project/requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY ml_project/utils app/ml_project/utils
COPY ml_project/models app/ml_project/models

COPY config.py app/config.py
COPY server.py app/server.py


WORKDIR /app
CMD ["python", "server.py"]