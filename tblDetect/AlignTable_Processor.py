from copy import deepcopy
import cv2
import numpy as np
from PIL import Image
import math
from lineVision.lineCv.processor.Cv_Line_Processor import Cv_Line_Processor
from lineVision.lineCv.line.LineCv_Line import LineCv_Line
from lineVision.LineCvUtils import LineCvUtils

from lineVision.DocumentBbZv import DocumentBBZv
from typing import List

class AlignTable_Processor:
    ################################################################
    def __init__(self, img_pil, annotation=None, tblBox=None):
        assert tblBox is not None or annotation is not None, "tblBox and annotation cannot both be none"
        self.img_pil = img_pil
        self.annotation = annotation
        self.tblBox = tblBox
        self.mask = None
        self.angle = None
        if annotation is not None:
            self.setMaskFromAnnotation(annotation)
            
    ################################################################
    def setMaskFromAnnotation(self, annotation):
        m = annotation.bool()
        m=m.cpu().numpy()
        w,h = self.img_pil.size
        mask = np.zeros((h, w, 1), np.uint8)
        mask[m] = 255
        self.mask = mask
        return mask
    
    ################################################################
    def boundToImgSize(self,x,y,x1,y1):
        w,h = self.img_pil.size
        x = max(x,0)
        y = max(y,0)
        x1 = min(x1,w-1)
        y1 = min(y1,h-1)
        return x,y,x1,y1        
     
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
    def unRotateAllCell(self,cells):
        cells = deepcopy(cells)
        for c in cells:
            rotated_bbox = self.unRotateCell(c)
            c["bbox"] = rotated_bbox.tolist()
        return cells
        
    ################################################################
    def unRotateCell(self,cell):
        # calculate bbox in original image
        (x1,y1,x2,y2) = cell["bbox"]
        bbox4 = np.array([(x1,y1),(x2,y1),(x2,y2),(x1,y2)])
        xBias, yBias, _, _ = self.cropBBox
        bbox4[:,0] += xBias
        bbox4[:,1] += yBias
        
        if self.angle is None:
            return np.intp(bbox4)
        # un-rotate box
        rotated_bbox = np.array([self.rotate_point(pt) for pt in bbox4])
        
        return rotated_bbox
        
         
    ################################################################
    def rotate_point(self, point, center=None, angle=None):
        """
        Calculates the new location of a point after rotation around a center.

        Args:
            center: A tuple (x, y) representing the center of rotation.
            angle: The rotation angle in degrees.
            point: A tuple (x, y) representing the point to be rotated.

        Returns:
            A tuple (x, y) representing the new location of the point.
        """
        if center is None:
            center = self.center
        if angle is None:
            angle = self.angle

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
    
   
    ################################################################
    def findLines(self, img_patch):
        documentBBZv = DocumentBBZv(img_patch)
        self.cvProcessor = cvProcessor = Cv_Line_Processor(
            documentBBZv=documentBBZv,
            )   
        cvProcessor.process()     
        self.lines = lines = cvProcessor.finalLine_Processor.getLines()
        lines = [l for l in lines if l.isHorizontal()]
        lines.sort(key=lambda l: -len(l)) 
        
        self.lines = lines
        return lines     
    ################################################################
    def getLinePixels(self, line: LineCv_Line):
        pixels = []
        for ls in line:
            left, bottom, right, top = ls.boundingBox
            if ls.patch is not None:
                for y in range(bottom, top):
                    for x in range(left, right):
                        if ls.patch[y-bottom, x-left] > 0:
                            pixels.append([x,y])
            else: 
                pixels.append([left,bottom])
        return pixels    
    
    ################################################################
    def getCropBBox(self):

        contour = self.getTblContour()
        bRect = cv2.boundingRect(contour)
        l,b,w,h = bRect
        r,t = l+w,b+h   
        intersect = LineCvUtils.calcBBIntersection((l,b,r,t), self.tblBox)
        if intersect[0] >= 0.9:
            self.cropBBox = self.tblBox
        else:
            self.cropBBox = self.boundToImgSize(l,b,r,t)

        self.center = (l+r)//2,(b+t)//2
        return self.cropBBox     
        
    ################################################################
    def find_approximate_line_from_pixels(self,points_in_line):
        """
        Finds the approximate line using linear regression for a given set of points.

        Args:
            points_inside: A list of tuples representing the points to fit the line to.

        Returns:
            A tuple containing the slope and intercept of the estimated line.
        """

        x_vals = [point[0] for point in points_in_line]
        y_vals = [point[1] for point in points_in_line]

        # Use numpy's polyfit function for linear regression
        slope, intercept = np.polyfit(x_vals, y_vals, 1)
        self.slope = slope
        self.intercept = intercept
        return slope, intercept    
    ################################################################
        
    def calculate_angle(self, slope):
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
        self.angle = angle_in_degrees = np.rad2deg(radians)

        return angle_in_degrees    
    
    ################################################################

    def find_approximate_line(self, line):
        linePixels = self.getLinePixels(line)
        slope, _ = self.find_approximate_line_from_pixels(linePixels)
        return slope
        
    ################################################################
    def getRotateAngleFromLines(self, lines):
        slopes = []
        for line in lines[:2]:
            if len(line) > 6:
                slope = self.find_approximate_line(line)
                # if abs(slope) < 0.001:
                #     return 0
                slopes.append(slope)
        if len(slopes) == 0:
            return 0
        # Calculate standard deviation using numpy.std
        standard_deviation = np.std(slopes)
        # if standard_deviation > 0.1:
        #     return 0
        
        mean = np.mean(slopes)
        print( "mean", mean, "std",standard_deviation, "slopes[0]",slopes[0])
        # slopes_filtered = [v for v in slopes if abs(v)-mean <=standard_deviation]
        # mean = np.mean(slopes_filtered)
        angle = self.calculate_angle(mean)
        return angle
        
    ################################################################
    def getAlignTable(self):
        l,b,r,t = self.getCropBBox()
        img = np.array(self.img_pil)
        tbl_patch = img[b:t, l:r]
        lines = self.findLines(tbl_patch)
        if len(lines) > 0:
            self.angle = angle = self.getRotateAngleFromLines(lines)
            if angle != 0:
                # rotate
                imgRotated =  self.img_pil.rotate(angle, 
                                                center=self.center, 
                                                resample=Image.BILINEAR,fillcolor=(255, 255, 255))
        
                tbl_patch = np.array(imgRotated)
                self.tbl_patch = tbl_patch = tbl_patch[b:t, l:r]
            
        tbl_patch_pil = Image.fromarray(tbl_patch)
        return tbl_patch_pil 



   
