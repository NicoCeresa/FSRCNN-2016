"""
FSRCNN Image Super-Resolution Web Application
Provides REST API endpoints for upscaling images using FSRCNN models
"""

import os
import time
import logging
from pathlib import Path

import torch
import numpy as np
from flask import Flask, request, render_template, send_from_directory, url_for
from torchvision.transforms.functional import to_pil_image
from PIL import Image

from utilities import upscale, FSRCNN


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

UPLOAD_FOLDER = 'api/static/input/'
OUTPUT_FOLDER = 'api/static/output/'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f"Using device: {device}")


def load_model(scale: int, model_path: str) -> torch.nn.Module:
    model = FSRCNN(scale=scale).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model = torch.compile(model, mode='reduce-overhead')
    logger.info(f"Loaded X{scale} model (compiled)")
    return model


model_x2 = load_model(2, 'api/models/FSRCNN_2s_10e_1b_0.2.0.pth')
model_x3 = load_model(3, 'api/models/FSRCNN_3s_10e_1b_0.2.0.pth')
model_x4 = load_model(4, 'api/models/FSRCNN_4s_10e_1b_0.2.0.pth')

MODELS = {
    2: model_x2,
    3: model_x3,
    4: model_x4
}


def get_requested_scales(form_data) -> list[int]:
    scales = []
    if form_data.get('scale_x2'):
        scales.append(2)
    if form_data.get('scale_x3'):
        scales.append(3)
    if form_data.get('scale_x4'):
        scales.append(4)
    
    return scales if scales else [2, 3, 4]


def process_image_upload(image_file) -> tuple[str, str, str]:
    filename_base = Path(image_file.filename).stem
    image_location = os.path.join(UPLOAD_FOLDER, image_file.filename)
    
    if not os.path.exists(image_location):
        image_file.save(image_location)
        logger.info(f"Saved image to: {image_location}")
    
    return image_location, filename_base, image_file.filename


def upscale_multiple_scales(image_location: str, filename_base: str, scales: list[int]) -> dict[int, float]:
    processing_times = {}
    
    for scale in scales:
        scale_start = time.time()
        output_path = f'X{scale}/X{scale}_{filename_base}.jpg'
        upscale(MODELS[scale], image_location, output_path)
        
        processing_time = time.time() - scale_start
        processing_times[scale] = processing_time
        logger.info(f"X{scale} upscale completed in {processing_time:.2f}s")
    
    return processing_times


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/prediction', methods=['POST'])
def get_prediction():
    if request.method != 'POST':
        return render_template('index.html', image_name=None)
    
    start_time = time.time()
    image_file = request.files.get("image")
    
    if not image_file:
        logger.warning("No image file provided")
        return render_template('index.html', image_name=None)
    
    logger.info(f"Processing image: {image_file.filename}")
    
    image_location, filename_base, original_filename = process_image_upload(image_file)
    
    scales_requested = get_requested_scales(request.form)
    logger.info(f"Requested scales: {scales_requested}")
    
    processing_times = upscale_multiple_scales(image_location, filename_base, scales_requested)
    
    total_time = time.time() - start_time
    logger.info(f"Total processing time: {total_time:.2f}s")
    
    output_filename = f"{filename_base}.jpg"
    return render_template(
        'result.html',
        image_name=output_filename,
        original_name=original_filename,
        scales=scales_requested
    )


@app.route('/api/static/output/<path:filename>', methods=['GET'])
def download_img(filename):
    logger.info(f"Download requested: {filename}")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, 'static', 'output')
    
    logger.info(f"Serving from directory: {output_dir}")
    return send_from_directory(
        output_dir,
        filename,
        as_attachment=True,
        mimetype='image/jpeg'
    )


@app.route('/prediction/health-check', methods=['GET'])
def get_health_check():
    logger.info("Health check called")
    return 'ok'


if __name__ == '__main__':
    logger.info("Starting Flask app on 0.0.0.0:8000")
    app.run(host='0.0.0.0', port=8000, debug=True)
