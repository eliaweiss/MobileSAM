import cv2
import numpy as np
from PIL import Image
import math

class AlignTable_Processor:
    ################################################################
    def __init__(self, img, annotation=None):
        self.img = img
        if annotation is not None:
            self.setMaskFromAnnotation(annotation)
            
    ################################################################
    def setMaskFromAnnotation(self, annotation):
        m = annotation.bool()
        m=m.cpu().numpy()
        w,h = self.img.size
        mask = np.zeros((h, w, 1), np.uint8)
        mask[m] = 255
        self.mask = mask
        return mask
    
    ################################################################
    def getAlignTable(self):
        contour = self.getTblContour()
        center, (w,h), angle = cv2.minAreaRect(contour)
        self.minAreaRect = center, (w,h), angle
        minAreaBBox = np.int0(cv2.boxPoints(self.minAreaRect))
        bRect = cv2.boundingRect(minAreaBBox)
        x,y,w,h = bRect
        x1,y1 = x+w,y+h   
        self.cropBBox = [x,y,x1,y1]     
        imgRotated =  self.img.rotate((angle-90), center=center, resample=Image.BILINEAR,fillcolor=(255, 255, 255))
        imgRotated    
        tbl_patch = np.array(imgRotated)
        tbl_patch = tbl_patch[y:y1, x:x1]
        tbl_patch_pil = Image.fromarray(tbl_patch)
        return tbl_patch_pil            
        
    ################################################################
    def getTblContour(self):
        contours, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = None
        rectArea = None
        for c in contours:
            _,_,h,w =cv2.boundingRect(c)
            if rectArea is None or rectArea < h*w:
                rectArea = h*w
                contour = c
        self.contour = contour
        return contour
        
                
                
    ################################################################
    def rotate_point(center, angle, point):
        """
        Calculates the new location of a point after rotation around a center.

        Args:
            center: A tuple (x, y) representing the center of rotation.
            angle: The rotation angle in degrees.
            point: A tuple (x, y) representing the point to be rotated.

        Returns:
            A tuple (x, y) representing the new location of the point.
        """

        # Convert angle to radians
        radians = math.radians(angle)

        # Translate point relative to center
        translated_point = (point[0] - center[0], point[1] - center[1])

        # Apply rotation matrix
        new_x = translated_point[0] * math.cos(radians) - translated_point[1] * math.sin(radians)
        new_y = translated_point[0] * math.sin(radians) + translated_point[1] * math.cos(radians)

        # Translate back to original coordinate system
        rotated_point = (int(new_x + center[0]), int(new_y + center[1]))

        return rotated_point                