FROM python:3.11

WORKDIR /app

COPY train /app/

COPY ./src/vectorizer.py /app

COPY ./scr/data_loader.py /app

COPY requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt

RUN python3 data_loader.py
RUN python3 train.py