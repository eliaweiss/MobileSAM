import numpy as np
import cv2

import base64
from PIL import ExifTags, Image


class FlaskUtil:

    ################################

    def resizeRotatedCells(rotated_cells, from_WxH, to_WxH):
        for cell in rotated_cells:
            bbox = cell['bbox']
            bbox = FlaskUtil.resizePoints(np.array(bbox), from_WxH, to_WxH)
            bbox = np.intp(bbox).tolist()
            cell['bbox'] = bbox
            
        return rotated_cells

    def base64_to_cv2Img(base64_string):
        # Decode base64 string to bytes
        image_bytes = base64.b64decode(base64_string)

        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)

        # Decode numpy array to image
        image_cv2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        return image_cv2

    ################################
    def base64_to_pil(base64_string):
        img = FlaskUtil.base64_to_cv2Img(base64_string)
        # cv2.imwrite("img.jpg",img)
        img_pil = Image.fromarray(img)
        return img_pil,img

    ##########################################

    def resizePoints(contour, from_WxH, to_WxH):

        orig_W, orig_H = to_WxH
        resized_W, resized_H = from_WxH

        # Calculate scale factors
        scale_x = orig_W / resized_W
        scale_y = orig_H / resized_H

        # Create a copy of the contour and scale its coordinates
        scaled_contour = contour.copy()
        scaled_contour[:, 0] = contour[:, 0] * \
            scale_x  # Scale x-coordinates
        scaled_contour[:, 1] = contour[:, 1] * \
            scale_y  # Scale y-coordinates

        return scaled_contour

    def boxToCtr(box):
        l,b,r,t = box
        ctr = np.array([[l,b],[r,b],[r,t], [l,t]])   
        return ctr
    
    def boxesToCtrs(boxes):
        return [FlaskUtil.boxToCtr(b) for b in boxes]