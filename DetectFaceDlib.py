# Face detection by Dlib

## Dlib installation steps
# 1. brew install boost
# 2. brew install boost-python
# 3. Install XQuartz-2.7.9
#    sudo ln -s /opt/X11/include/X11 /usr/local/include
# 4. Download dlib
#    cd dlib-18.18/
#    sudo python setup.py install
#
# Running samle
# a. pip install scikit-image
# b. Download https://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2
# c. python ./face_landmark_detection.py shape_predictor_68_face_landmarks.dat ../examples/faces

## reference
# - http://dlib.net/face_detector.py.html

import dlib
import cv2

predictor_path = "./dlib_data/shape_predictor_68_face_landmarks.dat"

class FaceDetector():
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)

    def detectFace(self, gray):
        dets = self.detector(gray, 1)
        if len(dets) == 0:
            return []
        # drop k (detected face index)
        return [(d.left(), d.top(), d.right(), d.bottom()) for k, d in enumerate(dets)]
    def detectParts(self, gray, x, y, x2, y2):
        eyes = self.predictor(gray, dlib.rectangle(x, y, x2, y2))
        return eyes
    def drawFacialParts(self, disp, parts):
        if parts.num_parts != 68:
            return
        p = parts.part
        idxs = [range(1, 16+1), range(28, 30+1), range(18, 21+1), range(23, 26+1),
                range(31, 35+1), range(37, 41+1), range(43, 47+1), range(49, 59+1),
                range(61-1, 67+1)]
        [cv2.line(disp, (p(i).x, p(i).y), (p(i-1).x,p(i-1).y), (100, 100, 255), 1)
         for r in idxs for i in r]
    def detectAndDraw(self, disp, frame):
        '''Detect face and eyes from the frame, and draw results on the disp.'''
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = frame
        faces = self.detectFace(gray)
        for (x, y, x2, y2) in faces:
            parts = self.detectParts(gray, x, y, x2, y2)
            #ps = [(parts.part(i).x, parts.part(i).y, parts.part(i+1).x, parts.part(i+1).y) \
            #      for i in range(0, parts.num_parts, 2)]
            #[cv2.line(disp, (p[0],p[1]), (p[2],p[3]), (100, 100, 255), 2) for p in ps]
            self.drawFacialParts(disp, parts)
            cv2.rectangle(disp, (x, y), (x2, y2), (0, 0, 255), 2)
            
