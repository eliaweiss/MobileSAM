import numpy as np
import torch

from tblDetect.TableDetect import TableDetect
from transformers import DetrForObjectDetection, DetrImageProcessor
from ultralyticsplus import YOLO

# global
tblDec = TableDetect()

# load model
model_yolov8 = YOLO('keremberke/yolov8m-table-extraction')

# set model parameters
model_yolov8.overrides['conf'] = 0.25  # NMS confidence threshold
model_yolov8.overrides['iou'] = 0.45  # NMS IoU threshold
model_yolov8.overrides['agnostic_nms'] = False  # NMS class-agnostic
model_yolov8.overrides['max_det'] = 1000  # maximum number of detections per image




# you can specify the revision tag if you don't want the timm dependency
processor_detr = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50") #, revision="no_timm")
model_detr = DetrForObjectDetection.from_pretrained("TahaDouaji/detr-doc-table-detection") #, revision="no_timm")


################################
def tblDecMicrosoft( image, threshold = 0.1):
    scores, boxes = tblDec.detectTables(image, threshold = threshold)
    return scores, boxes

################################
def model_yolov8_detect( image):
    boxes = []
    labels = []
    scores = []
    results = model_yolov8.predict(image)
    
    for res in results:
        boxes.extend(np.intp(res.boxes.xyxy).tolist())
        labels.extend(res.boxes.cls.tolist())
        scores.extend(res.boxes.conf.tolist())
    return scores, boxes, labels

################################
def tblDec_detr( image, threshold = 0.1):
    inputs = processor_detr(images=image, return_tensors="pt")
    outputs = model_detr(**inputs)

    # convert outputs (bounding boxes and class logits) to COCO API
    # let's only keep detections with score > 0.9
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor_detr.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=threshold)[0]
    return results["scores"].tolist(), np.intp(results["boxes"].tolist()).tolist()

class MultiDetectors:
    
    def multiDetector(image):
        scores, boxes = tblDecMicrosoft(image)
        scores_t, boxes_t,_ = model_yolov8_detect(image)
        scores.extend(scores_t)
        boxes.extend(boxes_t)
        scores_t, boxes_t = tblDec_detr(image)
        scores.extend(scores_t)
        boxes.extend(boxes_t)  
        return scores, boxes