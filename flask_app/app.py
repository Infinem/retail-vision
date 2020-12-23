import os
import base64

from flask import Flask
from flask import request, jsonify

import numpy as np
from PIL import Image

from yolov5 import yolov5_detection

ROOT_DIR = os.getcwd()

def encode_image(image_path):
    pil_img = Image.open(image_path, mode='r') 
    byte_arr = io.BytesIO()
    pil_img.save(byte_arr, format='JPEG')
    encoded_img = base64.encodebytes(byte_arr.getvalue()).decode('ascii')
    return encoded_img

def decode_image(image_string):
    image_bytes = image_string.encode("utf-8")
    image = base64.b64decode(image_bytes)
    return image

server = Flask(__name__)

@server.route("/", methods=["GET", "POST"])
def hello_world():
    return "The server is up and running. Send an image file."

@server.route("/detect", methods=["POST"])
def detect():
    if request.json:
        image_bytes = decode_image(request.json['image'])
        with open("./tmp/current.jpg", "wb") as image_file:
            image_file.write(image_bytes)
        yolov5_detection.detect("./tmp/current.jpg")
        encoded_result = encode_image("./tmp/result.jpg")
        return jsonify({"result": encoded_result})