
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
import torch


# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]


#########################################
def plot_rotatedResult(img, rotated_cells):
    tmpImg = np.array(img)
    pil_img = applyRotatedResult(rotated_cells, tmpImg)
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    plt.axis('off')
    plt.show() 

#########################################

def applyRotatedResult(tmpImg, rotated_cells ):
    for cell in rotated_cells:
        rotated_bbox = np.array(cell['bbox'])
        color = np.intp(np.random.random(3)*255).tolist()
        cv2.drawContours(tmpImg, [rotated_bbox], 0, color, 4)  
  
#########################################
def plot_results(pil_img, prob, boxes):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for score, (xmin, ymin, xmax, ymax), c in zip(prob, boxes, colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                fill=False, color=c, linewidth=3))
        text = f'{score:0.2f}'
        
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()
        
#########################################
def plot_annotations(img_pil, anns):
    imgTmp = np.zeros((anns.shape[1], anns.shape[2], 3), dtype=np.uint8)
    for ann in range(anns.shape[0]):
        m = anns[ann].bool()
        m=m.cpu().numpy()
        color_mask = np.int0(np.random.random(3)*255)
        imgTmp[m] = color_mask

    background =  img_pil.convert("RGBA")
    imgTmp = Image.fromarray(imgTmp).convert("RGBA")
    pil_img = Image.blend(background, imgTmp, 0.5)            
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    plt.axis('off')
    plt.show() 

#########################################
    
def plot_results_cells(tblStructDetect, tbl_patch, cells, class_to_visualize):
    if class_to_visualize not in tblStructDetect.id2label.values():
      raise ValueError(f"Class should be one of the available classes {tblStructDetect.id2label}")

    plt.figure(figsize=(16,10))
    plt.imshow(tbl_patch)
    ax = plt.gca()

    for cell in cells:
        bbox = cell["bbox"]
        label = cell["label"]

        if label == class_to_visualize:
          xmin, ymin, xmax, ymax = tuple(bbox)

          ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=np.random.random(3), linewidth=3))
          plt.axis('off')
              
################################################################
# for output bounding box post-processing
def box_cxcywh_to_xyxy( x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)    
        
################################################################
def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b
################################################################

def outputs_to_objects(outputs, img_size, id2label):
    m = outputs.logits.softmax(-1).max(-1)
    pred_labels = list(m.indices.detach().cpu().numpy())[0]
    pred_scores = list(m.values.detach().cpu().numpy())[0]
    pred_bboxes = outputs['pred_boxes'].detach().cpu()[0]
    pred_bboxes = [elem.tolist() for elem in rescale_bboxes(pred_bboxes, img_size)]

    objects = []
    for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
        class_label = id2label[int(label)]
        if not class_label == 'no object':
            objects.append({'label': class_label, 'score': float(score),
                            'bbox': [float(elem) for elem in bbox]})

    return objects   

def calculate_angle(slope):
    """
    Calculates the angle of the line relative to the x-axis in degrees.

    Args:
    slope: The slope of the estimated line.

    Returns:
    The angle of the line in degrees.
    """

    # Use arctangent (atan) to find the angle in radians
    radians = np.arctan(slope)

    # Convert radians to degrees
    angle_in_degrees = np.rad2deg(radians)

    return angle_in_degrees