import torchvision.transforms as T
import torch
torch.set_grad_enabled(False);

from transformers import AutoModelForObjectDetection
from lineVision.LineCvUtils import LineCvUtils

# from transformers import TableTransformerForObjectDetection


class TableDetect:
    def __init__(self):
        # standard PyTorch mean-std input image normalization
        self.transform = T.Compose([
            T.Resize(800),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.model = AutoModelForObjectDetection.from_pretrained("microsoft/table-transformer-detection", revision="no_timm")        
        # self.model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")
        self.model.eval();

    # for output bounding box post-processing
    def box_cxcywh_to_xyxy(self,x):
        x_c, y_c, w, h = x.unbind(1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
            (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)

    def rescale_bboxes(self,out_bbox, size):
        img_w, img_h = size
        b = self.box_cxcywh_to_xyxy(out_bbox)
        b = (b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)).type(torch.int)  
        return b        
        
      
    
    def detectTables(self,im, origSize = None):
        if origSize is None:
            origSize = im.size
        # mean-std normalize the input image (batch-size: 1)
        img = self.transform(im).unsqueeze(0)

        # propagate through the model
        outputs = self.model(img)
        probas = outputs['logits'].softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > 0.04
        probas = probas[keep]
        probas = probas[:, 0]
        
        # convert boxes from [0; 1] to image scales
        bboxes_scaled = self.rescale_bboxes(outputs['pred_boxes'][0, keep], origSize)   

        # Apply NMS to suppress overlapping bounding boxes
        indices = non_max_suppression(bboxes_scaled, probas, threshold=0.01)

        # Extract the final boxes after NMS
        bboxes_scaled = [bboxes_scaled[i].tolist() for i in indices]                             
        probas = [probas[i].tolist() for i in indices]
                
        return probas, bboxes_scaled 
    
    
def non_max_suppression(boxes, scores, threshold):
    """
    Perform Non-Maximum Suppression to remove overlapping bounding boxes.
    
    Args:
    - boxes: A list of bounding box coordinates in the format [x_min, y_min, x_max, y_max].
    - scores: A list of confidence scores corresponding to each bounding box.
    - threshold: IoU (Intersection over Union) threshold to determine overlapping boxes.
    
    Returns:
    - selected_indices: A list of indices corresponding to the selected bounding boxes.
    """
    selected_indices = []
    
    # Sort boxes based on their scores in descending order
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    
    while sorted_indices:
        # Select the bounding box with the highest score
        selected_index = sorted_indices[0]
        selected_indices.append(selected_index)
        
        # Compute IoU between the selected box and other boxes
        selected_box = boxes[selected_index]
        other_boxes = [boxes[i] for i in sorted_indices[1:]]
        ious = [calculate_iou(selected_box, other_box) for other_box in other_boxes]
        
        # Remove boxes with IoU greater than the threshold
        filtered_indices = [i for i, iou in enumerate(ious) if iou <= threshold]
        sorted_indices = [sorted_indices[i + 1] for i in filtered_indices]
    
    return selected_indices

def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) between two bounding boxes.
    
    Args:
    - box1, box2: Bounding box coordinates in the format [x_min, y_min, x_max, y_max].
    
    Returns:
    - iou: Intersection over Union (IoU) value.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - intersection
    
    iou = intersection / union if union > 0 else 0
    
    return iou
    