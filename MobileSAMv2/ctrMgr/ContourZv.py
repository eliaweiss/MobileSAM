
from typing import TYPE_CHECKING

import cv2

if TYPE_CHECKING:
    from MobileSAMv2.ctrMgr.CloseContourManager import CloseContourManager
    pass
    # from vision.contour.CloseContourManager import CloseContourManager


class ContourZv:
    def __init__(self, contourMgr: 'CloseContourManager', cntIndex, color, area = None, bRect= None) -> None:
        self.contourMgr = contourMgr
        self.cntIndex = cntIndex
        self.color = color
        self.area = area
        self.bRect =bRect
        self.bb = None
        
    ################################
    def getContour(self):
        return self.contourMgr.getRawContour(self.cntIndex)
    
    def getArea(self):
        if self.area is None:
            self.area = cv2.contourArea(self.getContour())
        return self.area
    

    def getBRect(self):
        if self.bRect is None:
            self.bRect = cv2.boundingRect(self.getContour())
        return self.bRect
            
    def getBB(self):
        if self.bb is None:
            l,b,w,h = self.getBRect()
            self.bb = l,b,l+w,b+h
        return self.bb