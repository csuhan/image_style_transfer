import os
import time

import cv2
import torch
from flask import Flask, flash, redirect, request, send_from_directory
from werkzeug.utils import secure_filename

from base_model import TransformNet
from utils import read_image, recover_image

cwd = os.getcwd()

model_path = os.path.join(cwd, 'models/sunflowers.pth')
model = TransformNet(16)
model.load_state_dict(torch.load(model_path))


ALLOWED_EXTENSIONS = {'png', 'jpg', 'bmp', 'jpeg'}
UPLOAD_FOLDER = os.path.join(cwd, 'uploads')
OUTPUT_FOLDER = os.path.join(cwd, 'outputs')
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_style(img_path):
    im = read_image(img_path)
    out = model(im)
    out = recover_image(out)
    out_filename = str(int(time.time()*1000))+'.jpg'
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, out_filename), out)
    return out_filename


@app.route('/style', methods=['POST'])
def hello():
    if 'file' not in request.files:
        flash('NO FILE PART')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('NO FILE SELECTED')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        out_filename = get_style(filepath)
        return redirect('/imgs/'+out_filename)


@app.route('/imgs/<path:path>')
def send_img(path):
    return send_from_directory(OUTPUT_FOLDER, path)
