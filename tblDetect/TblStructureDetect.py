import numpy as np
from PIL import Image
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_grad_enabled(False);
from torchvision import transforms
from transformers import TableTransformerForObjectDetection
from transformers import DetrFeatureExtractor


class MaxResize(object):
    def __init__(self, max_size=800):
        self.max_size = max_size

    def __call__(self, image):
        width, height = image.size
        current_max_size = max(width, height)
        scale = self.max_size / current_max_size
        resized_image = image.resize(
            (int(round(scale*width)), int(round(scale*height))))

        return resized_image


class TblStructureDetect:
    def __init__(self):
        self.structure_transform = transforms.Compose([
            MaxResize(1000),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # new v1.1 checkpoints require no timm anymore
        self.structure_model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-structure-recognition-v1.1-all")
        self.structure_model.to(device)
        # self.feature_extractor = DetrFeatureExtractor()
        # update id2label to include "no object"
        self.id2label = self.structure_model.config.id2label
        self.id2label[len(self.id2label)] = "no object"        
        
    ################################################################
    # for output bounding box post-processing
    def box_cxcywh_to_xyxy(self, x):
        x_c, y_c, w, h = x.unbind(-1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)    
    
    ################################################################
    def rescale_bboxes(self, out_bbox, size):
        img_w, img_h = size
        b = self.box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        return b
    ################################################################
        
    def outputs_to_objects(self, outputs, img_size):
        m = outputs.logits.softmax(-1).max(-1)
        pred_labels = list(m.indices.detach().cpu().numpy())[0]
        pred_scores = list(m.values.detach().cpu().numpy())[0]
        pred_bboxes = outputs['pred_boxes'].detach().cpu()[0]
        pred_bboxes = [elem.tolist() for elem in self.rescale_bboxes(pred_bboxes, img_size)]

        objects = []
        for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
            class_label = self.id2label[int(label)]
            if not class_label == 'no object':
                objects.append({'label': class_label, 'score': float(score),
                                'bbox': [float(elem) for elem in bbox]})

        return objects   
         
    ################################################################
    def detectTableStructure(self,tbl_patch):
        # encoding = self.feature_extractor(tbl_patch, return_tensors="pt")
        pixel_values = self.structure_transform(tbl_patch).unsqueeze(0)
        pixel_values = pixel_values.to(device)      
        # forward pass
        with torch.no_grad():
            outputs = self.structure_model(pixel_values)          

        cells = self.outputs_to_objects(outputs, tbl_patch.size)
        return cells
    
    