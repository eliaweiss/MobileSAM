# from tools.baseTools import BaseTools
# from tools.cleanImgTxt import CleanImgTxt
# from tools.isLineBetweenWords import IsLineBetweenWords
# from vision.impl.PaycheckZvImpl import PaycheckZvImpl
import cv2
from flask import Flask
from flask import Flask, request, jsonify, make_response, flash, send_file
from flask_cors import CORS
import os
from pathlib import Path
import base64
import numpy as np
import requests
import json
import os
from pathlib import PurePath
from PIL import Image, ExifTags
import io
import time
from os import environ
import boto3


import sys

from tblDetect.TableDetect import TableDetect


class FileRedirector:
    def __init__(self, file_path):
        self.file = open(file_path, 'a')

    def write(self, data):
        self.file.write(data)
        self.file.flush()  # Automatically flush after every write

    def flush(self):
        self.file.flush()




app = Flask(__name__)
# if not 'UPLOAD_FOLDER' in environ:
#     app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# else:
#     app.config['UPLOAD_FOLDER'] = environ['UPLOAD_FOLDER']

CORS(app)

@app.route('/')
def home():
    return "Hello, World!"


@app.route("/detectTbl", methods=["POST", "GET", "OPTIONS"])
def detectTbl():  
    # Assuming you're receiving JSON data
    request_data = request.json
    
    # Accessing the value of the 'tst' key in the JSON data
    base64_string = request_data.get('image')
    img_pil = base64_to_pil(base64_string)
    origSize = img_pil.size
    # Resize the image
    img_pil.thumbnail((1000, 1000)) 
    print("rescale",img_pil.size, "origSize",origSize)   
    tblDec = TableDetect()
    probas, boxes = tblDec.detectTables(img_pil, origSize=origSize)  
    res = {
        "detectTbl": [],
    }
    for score,bb in list(zip(probas, boxes)):
        bb = np.intp(bb).tolist()
        res['detectTbl'].append({
            "score": round(score, 2),
            "bbox": bb
        })
    
    # Return a response if needed
    return json.dumps(res)



if __name__ == '__main__':
    app.run(debug=True)