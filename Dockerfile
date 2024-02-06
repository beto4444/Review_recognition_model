FROM python:3.11

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/nltk_downloader.py .
RUN python nltk_downloader.py

COPY . .



# Ustaw PYTHONPATH
ENV PYTHONPATH "${PYTHONPATH}:/app/src"


CMD ["sh", "-c", "python src/data_loader.py && python src/train/train.py && python src/inference/inference.py"]
