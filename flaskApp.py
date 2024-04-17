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
from notebooks.Utils import applyRotatedResult
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


################################################################
################################################################


@app.route("/detectTbl", methods=["POST", "GET", "OPTIONS"])
def detectTbl():  
    start = time.time()
    
    # Assuming you're receiving JSON data
    request_data = request.json

    # Accessing the value of the 'tst' key in the JSON data
    base64_string = request_data.get('image')
    img_pil, img = FlaskUtil.base64_to_pil(base64_string)
    origSize = img_pil.size
    # Resize the image
    img_pil.thumbnail((1000, 1000)) 
    print("rescale",img_pil.size, "origSize",origSize)   
    probas, boxes = tblDec.detectTables(img_pil) # , origSize=origSize

    
    extractTblStructure = request_data.get('extractTblStructure')
    if request_data.get('extractCtr') or extractTblStructure:
        ctrList = []
        tableCells = []
        anns = sam.process(img_pil,boxes)
        for ann, box, prob in zip(anns, boxes,probas):
            alignTable_processor = AlignTable_Processor(img_pil, annotation=ann, tblBox=box)
            ctr = alignTable_processor.getTblApproxCtr()
            ctrList.append(ctr)
            if extractTblStructure:
                tbl_patch_pil = alignTable_processor.getAlignTable()
                cells = tblStructDetect.detectTableStructure(tbl_patch_pil)
                rotated_cells =  alignTable_processor.unRotateAllCell(cells)  
                
                FlaskUtil.resizeRotatedCells(rotated_cells, img_pil.size, origSize)
                                        
                applyRotatedResult(img, rotated_cells)
                cv2.imwrite("img_cell.jpg", img)                 
                tableCells.append(rotated_cells)         
    else:
        ctrList = boxes
        tableCells = np.empty((len(boxes), 0), dtype=int)
            
    res = {
        "detectTbl": [],
    }        
    
    for score,bb,ctr,cells in list(zip(probas, boxes, ctrList,tableCells)):
        ctr = FlaskUtil.resizePoints(ctr, img_pil.size, origSize)
        ctr = np.intp(ctr).tolist()
        bb = FlaskUtil.resizePoints(np.array(bb).reshape(2,2), img_pil.size, origSize)
        bb = np.intp(bb.flatten()).tolist()
        res['detectTbl'].append({
            "score": round(score, 2),
            "bbox": bb,
            "ctr": ctr,
            "cells": cells,
        })
    
    print("------ detectTbl time: (s): %s" % round(time.time() - start, 2))
    
    return json.dumps(res)

        # cv2.drawContours(img, [ctr], 0, (0, 255, 0), 4)  # Green bounding box with thickness 2
        # cv2.imwrite("img_ctr.jpg",img)


################################################################
################################################################


@app.route("/detectTblStructure", methods=["POST", "GET", "OPTIONS"])
def detectTblStructure():  
    start = time.time()
    
    # Assuming you're receiving JSON data
    request_data = request.json

    # Accessing the value of the 'tst' key in the JSON data
    base64_string = request_data.get('image')
    img_pil, img = FlaskUtil.base64_to_pil(base64_string)
    
    box = request_data.get('box' )
    if box is None:
        w,h = img_pil.size
        box = [0,0,w,h]
  
            
    alignTable_processor = AlignTable_Processor(img_pil, tblBox=box)

    tbl_patch_pil = alignTable_processor.getAlignTable()
    cells = tblStructDetect.detectTableStructure(tbl_patch_pil)
    rotated_cells =  alignTable_processor.unRotateAllCell(cells)   
    # applyRotatedResult(img, rotated_cells)
    # cv2.imwrite("img_cell.jpg", img)
            
    res = {
        "tableCells": rotated_cells,
    }        
    
    print("------ detectTableStructure time: (s): %s" % round(time.time() - start, 2))
    
    return json.dumps(res)

################################################################
################################################################


@app.route("/segment", methods=["POST", "GET", "OPTIONS"])
def segment():  
    start = time.time()
 
    # Assuming you're receiving JSON data
    request_data = request.json

    # Accessing the value of the 'tst' key in the JSON data
    base64_string = request_data.get('image')
    img_pil, img = FlaskUtil.base64_to_pil(base64_string)
    origSize = img_pil.size
    # Resize the image
    img_pil.thumbnail((1000, 1000)) 
    print("rescale",img_pil.size, "origSize",origSize)   
    
    boxes = request_data.get('boxes' )
    if boxes is None:
        w,h = img_pil.size
        boxes = [[0,0,w,h]]
    else:
        boxes = FlaskUtil.resizePoints(np.array(boxes).reshape(2,2), origSize, img_pil.size)
        boxes = np.intp(boxes.flatten()).tolist()        
    

    ctrList = []
    anns = sam.process(img_pil,boxes)
    for ann, box in zip(anns, boxes):
        alignTable_processor = AlignTable_Processor(img_pil, annotation=ann, tblBox=box)
        ctr = alignTable_processor.getApproxCtr()
        ctr = FlaskUtil.resizePoints(ctr, img_pil.size, origSize)
        # cv2.drawContours(img, [ctr], 0, (0, 255, 0), 4)  # Green bounding box with thickness 2
        # cv2.imwrite("img_ctr.jpg",img)        
        ctr = np.intp(ctr).tolist()
        ctrList.append(ctr)
        
    print("------ segment time: (s): %s" % round(time.time() - start, 2))
        
    res = {
        "ctrList": ctrList,
    }
    return json.dumps(res)

if __name__ == '__main__':
    app.run(debug=True)