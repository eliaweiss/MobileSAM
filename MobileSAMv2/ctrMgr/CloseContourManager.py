

import random
import cv2
import numpy as np
from scipy import stats


from typing import TYPE_CHECKING, List

from MobileSAMv2.ctrMgr.ContourZv import ContourZv
if TYPE_CHECKING:
    pass

def create_random_color():
    rgb = tuple(random.randint(0, 255) for _ in range(3))
    return rgb
class CloseContourManager:
    
    def __init__(self, image, minArea = 3000):
        self.minArea = minArea
        
        self.image = image
        self.height, self.width, _ = self.image.shape
        
        self._process()
        
    ################################
    def _process(self):
        image = self.image
        contourImage = np.ones([self.height, self.width, 3], dtype=np.uint8) * 255

        # Perform Canny to create a binary image for contour detection
        thresh = cv2.Canny(image, 75, 100, apertureSize=3)
        
        # # dilates
        thresh = cv2.dilate(thresh, np.ones((3, 3), np.uint8), iterations=1)
        # # erodes
        # thresh = cv2.erode(thresh, np.ones((3, 3), np.uint8), iterations=1)

        # Find the contours in the binary image
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        self.contours = contours
        self.hierarchy = hierarchy
        self.color2contour = {(255, 255, 255): None}
        self.contour2color = {}
        self.contourZvMap = {}
        h, w = image.shape[:2]
        contourMap = np.zeros((h, w), np.uint16)
        for i in range(len(contours)):
            hh = hierarchy[0, i]
            if hh[2] != -1:  # checks that current contour has children
            # if True:
                area = None
                area = cv2.contourArea(contours[i])
                if area < self.minArea:
                    continue
                # print("Internal Area:", area1)
                bRect = cv2.boundingRect(contours[i])
                _,_, w,h = bRect
                # if w*h < self.minArea: # or area > self.minArea:
                #     continue
                
                # choose unique color for each contour
                while True:
                    color = create_random_color()
                    if color not in self.color2contour:
                        break
                self.color2contour[color] = i
                self.contourZvMap[i] = ContourZv(self,i, color,area, bRect)
                self.contour2color[i] = color
                cv2.drawContours(contourImage, [contours[i]], -1,color,-1)
                # cv2.drawContours(contourMap, [contours[i]], -1,i,-1)
        self.contourMap = contourMap
        self.contourImage = contourImage 
        return contourImage     
    
    ################################################################
    def getContourByPoint(self, x,y):
            try:
                ctrIndex = self.contourMap[y,x]
                if ctrIndex is not None:
                    return self.contourZvMap[ctrIndex]
                    # return ContourZv(self, ctrIndex)
            except KeyError:
                pass
            return None
            
    ################################
    def getRawContour(self,i):
        return self.contours[i]            
    
    ################################################################
    def displayContourList(self, contours: List['ContourZv']):
        contourImage = np.ones([self.height, self.width, 3], dtype=np.uint8) * 255
        for contour in contours:
            contourVal = contour.getContour()
            if contourVal is not None:
                cv2.drawContours(contourImage, [contourVal], -1,contour.color,-1)        
            
        cv2.imshow('Close Ctr Result', contourImage)
        cv2.waitKey(0)   
    ################################################################
    def displayContour(self, contour: 'ContourZv'):
        self.display(contour.cntIndex)
    ################################################################
    def display(self, index= None):
        if index is None or index == 0:
            contourImage = self.contourImage
        else:
            contour = self.getRawContour(index)
            contourImage = np.ones([self.height, self.width, 3], dtype=np.uint8) * 255
            color = self.contour2color[index]
            cv2.drawContours(contourImage, [contour], -1,color,-1)        
        # Display the output image
        cv2.imshow('Close Ctr Result', contourImage)
        # cv2.waitKey(0)        
        
    ################################################################
    def destroyWindow(self):
        cv2.destroyWindow('Close Ctr Result')
        
    ################################################################
    def getContourByBox(self, box) -> 'ContourZv':
        l,b,r,t = box
        # boxCtr = self.contourMap[b:t,l:r]
        # ctrIndex = stats.mode(boxCtr.flatten())
        c1 = self.contourMap[b,l]
        c2 = self.contourMap[b,r]
        c3 = self.contourMap[t,l]
        c4 = self.contourMap[t,r]
        ctrIndex = stats.mode([c1,c2,c3,c4])
        try:
            if ctrIndex is not None and len(ctrIndex.mode) > 0 and ctrIndex.mode[0] != 0:
                ii = ctrIndex.mode[0]
                return self.contourZvMap[ii]
                # return ContourZv(self, ii)
        except KeyError:
            pass
        return None        
        