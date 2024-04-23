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
import uuid


import sys

from flaskUtil.FlaskUtil import FlaskUtil
from notebooks.Utils import applyRotatedResult
from tblDetect.AlignTable_Processor import AlignTable_Processor
from tblDetect.MobileSamBoxes import MobileSamBoxes
from tblDetect.TableDetect import TableDetect
from tblDetect.TblStructureDetect import TblStructureDetect
from lineVision.LineCvUtils import LineCvUtils

app = Flask(__name__)

CORS(app)

# global
IMAGE_SIZE =1500
sam = MobileSamBoxes()
tblDec = TableDetect()
tblStructDetect = TblStructureDetect()       
from ultralyticsplus import YOLO, render_result

# load model
model_yolov8 = YOLO('keremberke/yolov8m-table-extraction')

# set model parameters
model_yolov8.overrides['conf'] = 0.25  # NMS confidence threshold
model_yolov8.overrides['iou'] = 0.45  # NMS IoU threshold
model_yolov8.overrides['agnostic_nms'] = False  # NMS class-agnostic
model_yolov8.overrides['max_det'] = 1000  # maximum number of detections per image

from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import requests



# you can specify the revision tag if you don't want the timm dependency
processor_detr = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50") #, revision="no_timm")
model_detr = DetrForObjectDetection.from_pretrained("TahaDouaji/detr-doc-table-detection") #, revision="no_timm")

@app.route('/')
def home():
    return "Hello, World!"

def model_yolov8_detect(image):
    boxes = []
    labels = []
    scores = []
    results = model_yolov8.predict(image)
    
    for res in results:
        boxes.extend(np.intp(res.boxes.xyxy).tolist())
        labels.extend(res.boxes.cls.tolist())
        scores.extend(res.boxes.conf.tolist())
    return scores, boxes, labels

def tblDec_detr(image, threshold = 0.65):
    inputs = processor_detr(images=image, return_tensors="pt")
    outputs = model_detr(**inputs)

    # convert outputs (bounding boxes and class logits) to COCO API
    # let's only keep detections with score > 0.9
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor_detr.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=threshold)[0]
    return results["scores"].tolist(), np.intp(results["boxes"].tolist()).tolist()

@app.route("/detectTbl", methods=["POST", "GET", "OPTIONS"])
def detectTbl():  
    start = time.time()
    
    # Assuming you're receiving JSON data
    request_data = request.json

    # Accessing the value of the 'tst' key in the JSON data
    base64_string = request_data.get('image')
    img = FlaskUtil.base64_to_cv2Img(base64_string)
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    # probas, boxes,_ = model_yolov8_detect(img) # , origSize=origSize
    probas, boxes = tblDec_detr(img_pil) # , origSize=origSize
            
    res = {
        "detectTbl": [],
    }        
    
    for score,bb in list(zip(probas, boxes)):
        res['detectTbl'].append({
            "score": round(score, 2),
            "bbox": bb,
        })
    
    print("------ detectTbl time: (s): %s" % round(time.time() - start, 2))
    
    return json.dumps(res)


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
    img_pil.thumbnail((IMAGE_SIZE, IMAGE_SIZE)) 
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