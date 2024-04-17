import numpy as np
import cv2

import base64
from PIL import ExifTags, Image


class FlaskUtil:
     
    ################################
    def base64_to_cv2(base64_string):
        # Decode base64 string to bytes
        image_bytes = base64.b64decode(base64_string)
        
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        
        # Decode numpy array to image
        image_cv2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        return image_cv2
     
    ################################
    def base64_to_pil(base64_string):
        img = FlaskUtil.base64_to_cv2(base64_string)
        cv2.imwrite("img.jpg",img)
        img_pil = Image.fromarray(img)
        return img_pil     
     