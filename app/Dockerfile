FROM python:3.11.5-slim

WORKDIR /app
COPY . /app
RUN apt-get update

RUN pip install os opencv-python pickle numpy pathlib tqdm
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install -r requirements.txt

EXPOSE 5001
CMD ["python3","api/app.py"]