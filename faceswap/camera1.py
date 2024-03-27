import cv2
import keyboard
import sys
from imutils import face_utils
import numpy as np
import os
import cv2
import argparse

from face_detection import select_face, select_all_faces
from face_swap import face_swap

dst = "background.jpg"
out = "output.jpg"
cam = cv2.VideoCapture(0)

def face_recognition(src):
    parser = argparse.ArgumentParser(description='FaceSwapApp')
    parser.add_argument('--warp_2d', default=False, action='store_true', help='2d or 3d warp')
    parser.add_argument('--correct_color', default=False, action='store_true', help='Correct color')
    parser.add_argument('--no_debug_window', default=False, action='store_true', help='Don\'t show debug window')
    args = parser.parse_args()
    
    src_img = cv2.imread(src)
    dst_img = cv2.imread(dst)
    src_points, src_shape, src_face = select_face(src_img)
    cv2.imshow("source", src_face)
    dst_faceBoxes = select_all_faces(dst_img)
    if dst_faceBoxes is None:
        print('Detect 0 Face !!!')
        exit(-1)

    output = dst_img
    for k, dst_face in dst_faceBoxes.items():
        output = face_swap(src_face, dst_face["face"], src_points,
                           dst_face["points"], dst_face["shape"],
                           output, args)

    dir_path = os.path.dirname(out)
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

    cv2.imwrite(out, output)

    ##For debug
    if not args.no_debug_window:
        cv2.imshow("From", dst_img)
        cv2.imshow("To", output)
        cv2.waitKey(0)
        
        cv2.destroyAllWindows()
            
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