import cv2
import keyboard
import sys
from imutils import face_utils
import imutils
import numpy as np
import collections
import dlib
import os
import cv2
import argparse

from face_detection import select_face, select_all_faces
from face_swap import face_swap

dst = "background.jpg"
out = "output.jpg"
cam = cv2.VideoCapture(0)

def face_recognition(src):
    src_img = cv2.imread(src)
    dst_img = cv2.imread(dst)
            


while(True):
    ret,frame = cam.read()
    if ret:
        cv2.imshow("camera", frame)
        cv2.imwrite("camera.jpg", frame)
        cv2.waitKey(1)
        if keyboard.is_pressed("alt+p"):
        # if video is still left continue creating images
            src = "camera.jpg"
            cv2.imwrite(src, frame)
            face_recognition(src)
            break
    else:
        break

cam.release()
cv2.destroyAllWindows()