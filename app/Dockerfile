FROM python:3.11.5

RUN pip install os opencv-python pickle numpy pathlib tqdm
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

COPY ./api /

CMD ["python","./app/api/utilities.py"]