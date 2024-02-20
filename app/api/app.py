import os
import torch
import numpy as np
from flask import Flask, request, render_template, send_from_directory, url_for
from torchvision.transforms.functional import to_pil_image
from PIL import Image
from pathlib import Path
import base64
import uuid
import cv2
from utilities import upscale, FSRCNN

app = Flask(__name__)

model = FSRCNN(scale=3)
model.load_state_dict(torch.load(f='api/models/FSRCNN_3s_10e_1b_0.2.0.pth'))

UPLOAD_FOLDER = 'api/static/input/'
OUTPUT_FOLDER = 'static/output/'

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route("/", methods=['GET', 'POST'])
def upload_predict():
    return render_template('index.html', image_name=None)


@app.route('/prediction', methods=['GET', 'POST'])
def get_prediction():
    if request.method == 'POST':
        image_file = request.files["image"]
        if image_file:
            image_location = os.path.join(UPLOAD_FOLDER, image_file.filename)
            if not os.path.exists(image_location):
                image_file.save(image_location)
                output_location_x3 = os.path.join(OUTPUT_FOLDER, f'X3_{image_file.filename}')
                output3 = upscale(model, image_location, output_location_x3)
                output3_img = to_pil_image(output3)
                output3_img.save(output_location_x3)
            return render_template('result.html', image_name=image_file.filename)

    return render_template('index.html', image_name=None)


@app.route('/api/static/output/<string:filename>', methods=['GET', 'POST'])
def download_img(filename):
    return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=True)


@app.route('/prediction/health-check', methods=['GET'])
def get_health_check():
    return 'ok'


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)