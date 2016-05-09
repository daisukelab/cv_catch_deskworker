# This demonstrates how easily we could find moving area, for thinking how we could use this for further porpose...
# - Detect center of moving area and show it, using simple detection algorithm.
# - Detect moving area by thresholding magnitude of flow (move).

## class diagram
# http://www.yuml.me/diagram/scruffy/class/draw
# [IOpticalFlow]^[DenseOpticalFlow],[DenseOpticalFlow]^[DetectCenterImgByOF]

from __future__ import print_function
from OpticalFlowShowcase import *

class Range:
    def __init__(self, pmin=0, pmax=0):
        self.pmin, self.pmax = (pmin, pmax)
    def isValid(self):
        return self.pmin != self.pmax
    def set(self, pmin, pmax):
        self.pmin = pmin
        self.pmax = pmax

class DemoDetectCenterByOF(DenseOpticalFlow):
    def __init__(self):
        self.step = 16         # configure: pixel steps
        self.threshold = 1500  # configure: line sum of magnitude for move detection
        self.rowRange = Range()
        self.colRange = Range()
    def detectMovingArea(self, flow):
        # get flow magnitude for each points
        mag, _ = cv2.cartToPolar(flow[...,0], flow[...,1])
        # get col/row sum of lines
        cols = np.array([np.sum(row) for row in mag.T])
        rows = np.array([np.sum(col) for col in mag])
        # simple centroid search ==> finding max of sum
        centroid = (cols.argmax(), rows.argmax())
        std = (np.std(cols), np.std(rows))
        # row range
        movingCols = np.where(self.threshold <= cols)[0] # find valid lines
        movingRows = np.where(self.threshold <= rows)[0]
        isValid = 2 <= movingCols.shape[0] and 2 <= movingRows.shape[0] # valid if two or more lines found
        if isValid:
            colRange = Range(np.amin(movingCols), np.amax(movingCols))
            rowRange = Range(np.amin(movingRows), np.amax(movingRows))
        else:
            colRange = rowRange = Range()
        print("center=(%d, %d),\tcr=[%d, %d],\trr=[%d, %d],\tstd=(% .5d, % .5d)" % (centroid[0], centroid[1], colRange.pmin, colRange.pmax, rowRange.pmin, rowRange.pmax, std[0], std[1]), end="\n")
        return (centroid, colRange, rowRange, std)
        
    def makeResult(self, grayFrame, flow):
        # draw flow
        h, w = grayFrame.shape[:2]
        y, x = np.mgrid[self.step/2:h:self.step, self.step/2:w:self.step].reshape(2,-1)
        fx, fy = flow[y,x].T
        lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines + 0.5)
        vis = cv2.cvtColor(grayFrame, cv2.COLOR_GRAY2BGR)
        cv2.polylines(vis, lines, 0, (0, 255, 0))
        for (x1, y1), (x2, y2) in lines:
            cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)

        # draw detected centroid and moving area
        centroid, cr, rr, std = self.detectMovingArea(flow)
        cv2.circle(vis, (centroid[0], centroid[1]), 10, (255, 255, 0), -1)
        if cr.isValid():
            cv2.line(vis, (cr.pmin, h - 10), (cr.pmax, h - 10), (255, 100, 0), 10)
        if rr.isValid():
            cv2.line(vis, (w - 10, rr.pmin), (w - 10, rr.pmax), (255, 100, 0), 10)
        return vis

   
