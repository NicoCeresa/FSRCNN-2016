import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, request, render_template, send_from_directory, url_for
from torchvision.transforms.functional import to_pil_image
from PIL import Image
from pathlib import Path
import base64
import uuid
import cv2
from utilities import upscale, FSRCNN

app = Flask(__name__)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_x2 = FSRCNN(scale=2)
model_x2.load_state_dict(torch.load(f='api/models/FSRCNN_2s_10e_1b_0.2.0.pth', map_location=device))

model_x3 = FSRCNN(scale=3)
model_x3.load_state_dict(torch.load(f='api/models/FSRCNN_3s_10e_1b_0.2.0.pth', map_location=device))

model_x4 = FSRCNN(scale=4)
model_x4.load_state_dict(torch.load(f='api/models/FSRCNN_4s_10e_1b_0.2.0.pth', map_location=device))

UPLOAD_FOLDER = 'api/static/input/'
# OUTPUT_FOLDER = 'api/static/output/'

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
            output_format_x2 = f'X2/X2_{image_file.filename}'
            output_format_x3 = f'X3/X3_{image_file.filename}'
            output_format_x4 = f'X4/X4_{image_file.filename}'
            if not os.path.exists(image_location):
                image_file.save(image_location)
        
            # output_location_x3 = os.path.join(OUTPUT_FOLDER, 'X3_' + image_file.filename)
            tensor_x2 = upscale(model_x2, image_location, output_format_x2)
            tensor_x3 = upscale(model_x3, image_location, output_format_x3)
            tensor_x4 = upscale(model_x4, image_location, output_format_x4)
            # img_transform = to_pil_image(image_formatted)
            # img_transform.save(f"app/api/static/output{output_format}")
            return render_template('result.html', image_name=image_file.filename)

    return render_template('index.html', image_name=None)


@app.route('/api/static/output/<string:filename>', methods=['GET', 'POST'])
def download_img(filename):
    return send_from_directory('/static/output', filename, as_attachment=True)


@app.route('/prediction/health-check', methods=['GET'])
def get_health_check():
    return 'ok'


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)