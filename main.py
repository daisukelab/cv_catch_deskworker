# Better samples/python2/opt_flow.py

## reference
# - http://stackoverflow.com/questions/2601194/displaying-a-webcam-feed-using-opencv-and-python
# - http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html

import cv2
from DetectCenterOfMoveByOF import *
from DetectFace import *

usage_text = '''
Hit followings to switch to:
1 - Dense optical flow by HSV color image (default);
2 - Dense optical flow by lines;
3 - Dense optical flow by warped image;
4 - Lucas-Kande method.

Hit 's' to save image.

Hit 'f' to flip image horizontally.

Hit ESC to exit.
'''

def main():
    ## private routines
    def capture(vc):
        rval, frame = vc.read()
        if rval and flipImage:
            frame = cv2.flip(frame, 1)
        return (rval, frame)
        
    ## main starts here
    flipImage = True
    vc = cv2.VideoCapture(0)
    if not vc.isOpened():
        exit -1
            
    cv2.namedWindow("preview")
            
    ### try to get the first frame
    rval, frame = capture(vc)
    if not rval:
        exit -1
    of = DemoDetectCenterByOF()
    of.set1stFrame(frame)
    fd = FaceDetector()

    ### main work
    while rval:
        rval, frame = capture(vc)

        ### do it
        img = of.apply(frame)
        fd.detectAndDraw(img, frame)
        cv2.imshow("preview", img)

        ### key operation
        key = cv2.waitKey(1)
        if key == 27:         # exit on ESC
            print 'Closing...'
            break
        elif key == ord('s'):   # save
            cv2.imwrite('img_raw.png',frame)
            cv2.imwrite('img_w_flow.png',img)
            print "Saved raw frame as 'img_raw.png' and displayed as 'img_w_flow.png'"
        elif key == ord('f'):   # save
            flipImage = not flipImage
            print "Flip image: " + {True:"ON", False:"OFF"}.get(flipImage)

    ## finish
    vc.release()
    cv2.destroyWindow("preview")


if __name__ == '__main__':
    print usage_text
    main()
