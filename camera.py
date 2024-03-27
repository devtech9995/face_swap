import cv2
import keyboard
import sys
from imutils import face_utils
import imutils
import numpy as np
import collections
import dlib

def face_remap(shape):
    remapped_image = shape.copy()
   # left eye brow
    remapped_image[17] = shape[26]
    remapped_image[18] = shape[25]
    remapped_image[19] = shape[24]
    remapped_image[20] = shape[23]
    remapped_image[21] = shape[22]
   # right eye brow
    remapped_image[22] = shape[21]
    remapped_image[23] = shape[20]
    remapped_image[24] = shape[19]
    remapped_image[25] = shape[18]
    remapped_image[26] = shape[17]
   # neatening 
    remapped_image[27] = shape[0]

    remapped_image = cv2.convexHull(shape)
    return remapped_image

frame = cv2.imread("test2.jpg")
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.3,
    minNeighbors=3,
    minSize=(30, 30)
)

print("[INFO] Found {0} Faces!".format(len(faces)))

for (x, y, w, h) in faces:
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)
    
    roi_color = frame[y:y + h, x:x + w]
    print("[INFO] Object found. Saving locally.")
    cv2.imwrite(str(w) + str(h) + '_faces.png', roi_color)
    # image = cv2.imread("242242_faces.jpg")
    image = roi_color
    image = imutils.resize(image, width=100)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    out_face = np.zeros_like(image)

    # initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    # detector = dlib.get_frontal_face_detector()
    # predictor = dlib.shape_predictor(SHAPE_PREDICTOR)

    # detect faces in the grayscale image
    rects = detector(gray, 1)
    print(enumerate(rects))
    # loop over the face detections
    for (i, rect) in enumerate(rects):
    # """
    # Determine the facial landmarks for the face region, then convert the facial landmark (x, y)-coordinates to a NumPy array
    # """
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        #initialize mask array
        remapped_shape = np.zeros_like(shape) 
        feature_mask = np.zeros((image.shape[0], image.shape[1]))   

        # we extract the face
        remapped_shape = face_remap(shape)
        cv2.fillConvexPoly(feature_mask, remapped_shape[0:27], 1)
        feature_mask = feature_mask.astype(np.bool_)
        out_face[feature_mask] = image[feature_mask]
        cv2.imshow("mask_inv", out_face)
        cv2.imwrite("camera1.png", out_face)


status = cv2.imwrite('faces_detected.png', frame)
print("[INFO] Image faces_detected.png written to filesystem: ", status)