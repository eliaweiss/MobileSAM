
from typing import TYPE_CHECKING

import cv2

if TYPE_CHECKING:
    from MobileSAMv2.ctrMgr.CloseContourManager import CloseContourManager
    pass
    # from vision.contour.CloseContourManager import CloseContourManager


class ContourZv:
    def __init__(self, contourMgr: 'CloseContourManager', cntIndex, color) -> None:
        self.contourMgr = contourMgr
        self.cntIndex = cntIndex
        self.color = color
        self.area = None
        self.bb = None
        
    ################################
    def getContour(self):
        return self.contourMgr.getRawContour(self.cntIndex)
    
    def area(self):
        if self.area is None:
            self.area = cv2.contourArea(self.getContour())
        return self.area
    

    def getBB(self):
        if self.bb is None:
            rect = cv2.boundingRect(self.getContour())
            l,b,w,h = rect
            self.bb = l,b,l+w,b+h
        return self.bb