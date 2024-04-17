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

from flaskUtil.FlaskUtil import FlaskUtil
from tblDetect.AlignTable_Processor import AlignTable_Processor
from tblDetect.MobileSamBoxes import MobileSamBoxes
from tblDetect.TableDetect import TableDetect
from tblDetect.TblStructureDetect import TblStructureDetect


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

# global
sam = MobileSamBoxes()
tblDec = TableDetect()
tblStructDetect = TblStructureDetect()       

@app.route('/')
def home():
    return "Hello, World!"


@app.route("/detectTbl", methods=["POST", "GET", "OPTIONS"])
def detectTbl():  
    # Assuming you're receiving JSON data
    request_data = request.json

    # Accessing the value of the 'tst' key in the JSON data
    base64_string = request_data.get('image')
    img_pil, img = FlaskUtil.base64_to_pil(base64_string)
    origSize = img_pil.size
    img_pil_resize = img_pil.copy()
    # Resize the image
    img_pil_resize.thumbnail((1000, 1000)) 
    print("rescale",img_pil_resize.size, "origSize",origSize)   
    probas, boxes = tblDec.detectTables(img_pil_resize) # , origSize=origSize
    res = {
        "detectTbl": [],
    }
    
    extractTblStructure = request_data.get('extractTblStructure')
    if request_data.get('extractCtr') or extractTblStructure:
        ctrList = []
        tableCells = []
        anns = sam.process(img_pil_resize,boxes)
        for ann, box, prob in zip(anns, boxes,probas):
            alignTable_processor = AlignTable_Processor(img_pil_resize, annotation=ann, tblBox=box)
            ctr = alignTable_processor.getTblApproxCtr()
            ctr= ctr.squeeze()
            ctrList.append(ctr)
            tbl_patch_pil = alignTable_processor.getAlignTable()
            cells = tblStructDetect.detectTableStructure(tbl_patch_pil)
            rotated_cells =  alignTable_processor.unRotateAllCell(cells)   
            tableCells.append(rotated_cells)         
    else:
        ctrList = boxes
        
    if request_data.get('extractCtr'):
                
            
        
    
    for score,bb,ctr in list(zip(probas, boxes, ctrList)):
        ctr = FlaskUtil.resizePoints(ctr, img_pil_resize.size, img_pil.size)
        # cv2.drawContours(img, [ctr], 0, (0, 255, 0), 4)  # Green bounding box with thickness 2
        # cv2.imwrite("img_ctr.jpg",img)
        
        ctr = np.intp(ctr).tolist()
        bb = FlaskUtil.resizePoints(np.array(bb).reshape(2,2), img_pil_resize.size, img_pil.size)
        bb = np.intp(bb.flatten()).tolist()
        res['detectTbl'].append({
            "score": round(score, 2),
            "bbox": bb,
            "ctr": ctr,
        })
    
    # Return a response if needed
    return json.dumps(res)



if __name__ == '__main__':
    app.run(debug=True)