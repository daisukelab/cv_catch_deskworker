# Face detection sample

## reference
# - http://docs.opencv.org/3.1.0/d7/d8b/tutorial_py_face_detection.html#gsc.tab=0

import numpy as np
import cv2

class FaceDetector():
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier('opencv_data/haarcascade_frontalface_alt.xml')
        self.eye_cascade = cv2.CascadeClassifier('opencv_data/haarcascade_eye.xml')

    def detectFace(self, gray):
        rects = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4,
                                                   minSize=(30, 30), flags = cv2.CASCADE_SCALE_IMAGE)
        if len(rects) == 0:
            return []
        rects[:,2:] += rects[:,:2]
        return rects
    def detectEyes(self, gray, x, y, x2, y2):
        roi_gray = gray[y:y2, x:x2]
        eyes = self.eye_cascade.detectMultiScale(roi_gray)
        return eyes
    def detectAndDraw(self, disp, frame):
        '''Detect face and eyes from the frame, and draw results on the disp.'''
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detectFace(gray)
        for (x, y, x2, y2) in faces:
            eyes = self.detectEyes(gray, x, y, x2, y2)
            for (ex,ey,ew,eh) in eyes:
                roi_color = disp[y:y2, x:x2]
                cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (100, 100, 255), 2)
            cv2.rectangle(disp, (x, y), (x2, y2), (0, 0, 255), 2)
            
