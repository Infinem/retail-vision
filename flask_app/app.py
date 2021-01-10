import os
import base64

from flask import Flask
from flask import request, jsonify

import numpy as np
from PIL import Image

from yolov5 import yolov5_detection

ROOT_DIR = os.getcwd()

def encode_image(image_path):
   with open(image_path, "rb") as image_file:
      byte_content = image_file.read()
      encoded_img = base64.b64encode(byte_content)
      encoded_str = encoded_img.decode("utf-8")
      return encoded_str

def decode_image(image_string):
    image_bytes = image_string.encode("utf-8")
    image = base64.b64decode(image_bytes)
    return image

def clean_tmp():
    for f in os.listdir("./tmp"):
        os.remove(f"./tmp/{f}")

server = Flask(__name__)

@server.route("/", methods=["GET", "POST"])
def hello_world():
    return "The server is up and running. Send an image file."

@server.route("/detect", methods=["POST"])
def detect():
    data_dict = {}
    if request.args:
        data_dict["user_id"] = request.args["user_id"]
        data_dict["address"] = request.args["address"]
        data_dict["image"] = request.args["image"]
    elif request.form:
        data_dict["user_id"] = request.form["user_id"]
        data_dict["address"] = request.form["address"]
        data_dict["image"] = request.form["image"]
    elif request.json:
        data_dict["user_id"] = request.json["user_id"]
        data_dict["address"] = request.json["address"]
        data_dict["image"] = request.json["image"]
    else:
        return {"error": "empty request!"}

    user_id = data_dict["user_id"]
    address = data_dict["address"]
    image_bytes = decode_image(data_dict["image"])

    with open(f"./tmp/{user_id}.jpg", "wb") as image_file:
        image_file.write(image_bytes)
    try:
        yolov5_detection.detect(user_id, f"./tmp/{user_id}.jpg")
        encoded_result = encode_image(f"./tmp/result_{user_id}.jpg")
        clean_tmp()
    except:
        clean_tmp()
        return {"error": "nothing to detect!"}
    
    return jsonify({"user_id": user_id, "address": address, "result": encoded_result})
