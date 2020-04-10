'''
FOR TESTING MODEL
'''
import tensorflow as tf
import cv2
import numpy as np
import time
# from tensorflow.keras.backend import set_session
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
# from tensorflow.keras.tensorflow_backend import set_session
import os
import sys
import math
from tensorflow.python.client import device_lib
from scipy.spatial import distance
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-sh", "--input-shape", required=True,
                type=int, help="shape of network input")
ap.add_argument("--normalize", required=False,
                default=False,
                type=bool, help="is netowrk output normalized?")
ap.add_argument("-p", "--path", required=True, help="Path to model")
ap.add_argument("-v","--video-path", required=False,default="",help="Sample VDO for testing")
ap.add_argument("--use-gpu",required=False,default=False,type=bool,help="Use GPU for computation")
ap.add_argument("--show-vdo",default=False,required=False,help="Show video screen",type=bool)
args = vars(ap.parse_args())
show_vdo = args["show_vdo"]
path_to_model = args["path"]
input_shape = args["input_shape"]
normalize = args["normalize"]
path_to_vdo  = args["video_path"]
use_gpu = args["use_gpu"]
if use_gpu:
    device = '/gpu:0'
else:
    device = '/cpu:0'
print("LOADING MODEL {}".format(path_to_model))
with tf.device(device):
    model = load_model(path_to_model)  # load model
try:
    plot_model(model, to_file='model.png')  # model schematic
    model_image = cv2.imread("model.png")  # read model image
    mHeight, mWidth, _ = model_image.shape  # get shape of model image
    model_image = cv2.resize(
    model_image, (int(mWidth*(0.6)), int(mHeight*(0.6))))  # resize model image
    if show_vdo:
        cv2.imshow("model structure", model_image)  # show model structure
except:
    pass
net = cv2.dnn.readNetFromTensorflow(
    './opencv_face_detector_uint8.pb', './opencv_face_detector.pbtxt')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
if path_to_vdo == "":
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture(path_to_vdo)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def apply_clahe(frame):
    # -----Converting image to LAB Color model-----------------------------------
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    # -----Splitting the LAB image to different channels-------------------------
    l, a, b = cv2.split(lab)
    # -----Applying CLAHE to L-channel-------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    # -----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    limg = cv2.merge((cl, a, b))
    # -----Converting image from LAB Color model to RGB model--------------------
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final


def detect_points(face):
    start = time.time()
    me = np.array(face)/255
    x_test = np.expand_dims(me, axis=0)
    x_test = np.expand_dims(x_test, axis=3)
    with tf.device('/cpu:0'):  # use gpu for prediction
        y_test = model.predict(x_test)
    if normalize:
        label_points = (np.squeeze(y_test)*input_shape)+input_shape
    label_points = (np.squeeze(y_test))
    stop = time.time()
    print("Execution time:{}(s)".format(stop-start))
    return label_points

def calculate_mar(key_points):  # calculate mouth aspect ratio
    x_points = key_points[::2]
    y_points = key_points[1::2]
    coords = list(zip(x_points, y_points))


def calculate_mar(key_points):  # calculate mouth aspect ratio
    x_points = key_points[::2]
    y_points = key_points[1::2]
    coords = list(zip(x_points, y_points))
    # INNER MOUTH COORDINATES
    inner_89 = coords[89]
    inner_91 = coords[91]
    inner_95 = coords[95]
    inner_93 = coords[93]
    inner_88 = coords[88]
    inner_82 = coords[82]
    dis_89_95 = distance.euclidean(inner_89, inner_95)  # upper left
    dis_91_93 = distance.euclidean(inner_91, inner_93)  # upper right
    dis_82_88 = distance.euclidean(inner_82, inner_88)  # cross mouth
    MAR = (dis_89_95 + dis_91_93)/(2*dis_82_88)
    return MAR


def calculate_ear(key_points):  # calculate mouth aspect ratio
    x_points = key_points[::2]
    y_points = key_points[1::2]
    coords = list(zip(x_points, y_points))
    # LEFT EYE COORDINATES
    left_61 = coords[61]
    left_67 = coords[67]
    left_63 = coords[63]
    left_65 = coords[65]
    left_60 = coords[60]
    left_64 = coords[64]

    # RIGHT EYE COORDINATES
    right_69 = coords[69]
    right_75 = coords[75]
    right_71 = coords[71]
    right_74 = coords[74]
    right_68 = coords[68]
    right_72 = coords[72]

    p2_p6 = distance.euclidean(left_61, left_67)
    p3_p5 = distance.euclidean(left_63, left_65)
    p1_p4 = distance.euclidean(left_60, left_64)
    LEFT_EAR = (p2_p6+p3_p5)/(2*p1_p4)

    p2_p6 = distance.euclidean(right_69, right_75)
    p3_p5 = distance.euclidean(right_71, right_74)
    p1_p4 = distance.euclidean(right_68, right_72)
    RIGHT_EAR = (p2_p6+p3_p5)/(2*p1_p4)

    AVG_EAR = LEFT_EAR + RIGHT_EAR
    AVG_EAR /= 2

    return LEFT_EAR, RIGHT_EAR, AVG_EAR


def get_3d_direction(frame, start, stop, x_points, y_points):
    x1, y1 = start
    width, height = stop
    PADDING = 32
    min_y = min(y_points)
    max_y = max(y_points)
    # print(min_y, max_y)
    center_y = int((min_y+max_y)/2 - PADDING)
    # Y-AXIS coordinates
    Y_start_x = int(x1+width+PADDING/2)
    Y_start_y = center_y
    Y_stop_x = int(x1+width+PADDING/2)
    Y_stop_y = y_points[16]
    diff_y = Y_stop_y - Y_start_y
    diff_x = abs(Y_start_x - Y_stop_x)
    Y_stop_y = int(Y_start_y + diff_y*0.4)

    # X-AXIS coordinates
    X_start_x = int(x1+width+PADDING/2)
    X_start_y = center_y
    X_stop_x = x_points[28]
    X_stop_y = int((y_points[29]+y_points[28])/2)
    diff_x = X_stop_x - X_start_x
    X_stop_x = int(X_start_x + diff_x*0.5)

    # Z-AXIS coordinates
    Z_start_x = int(x1+width+PADDING/2)
    Z_start_y = center_y
    Z_stop_x = x_points[53]
    Z_stop_y = y_points[53]
    diff_x = Z_stop_x - Z_start_x
    Z_stop_x = int(Z_start_x + diff_x)
    # Draw arrow of each axis
    # Draw Y_AXIS
    cv2.arrowedLine(frame, (Y_start_x,
                            Y_start_y), (Y_stop_x, Y_stop_y), (0, 255, 0), 2)
    # Draw X_AXIS
    cv2.arrowedLine(frame, (X_start_x,
                            X_start_y), (X_stop_x, X_stop_y), (0, 0, 255), 2)
    # Draw Z_AXIS
    cv2.arrowedLine(frame, (Z_start_x, Z_start_y),
                    (Z_stop_x, Z_stop_y), (255, 0, 0), 2)

    direction_vector = np.array([Z_stop_x-Z_start_x, Z_stop_y-Z_start_y])
    direction_vector = direction_vector/np.linalg.norm(direction_vector)
    return direction_vector


def draw_face_estimator(frame, key_points, draw_index=False, draw_point=True, draw_contour=False):
    x_points = key_points[::2]
    y_points = key_points[1::2]
    f_height, f_width, _ = face.shape
    scale_x = f_width/input_shape
    scale_y = f_height/input_shape
    PADDING = 32
    x_points = np.array(x_points)*scale_x+x1
    x_points = x_points.astype(int)
    y_points = np.array(y_points)*scale_y+y1
    y_points = y_points.astype(int)
    coords = list(zip(x_points, y_points))
    for i, coord in enumerate(coords, start=0):
        x, y = coord
        COLOR = (255, 133, 71)
        if i in range(33, 51):  # face region
            COLOR = (0, 255, 0)
        elif i in range(51, 55):  # nose1
            COLOR = (25, 25, 25)
        elif i in range(55, 60):  # nose2
            COLOR = (0, 255, 255)
        elif i in range(60, 76):  # eye bounds
            COLOR = (255, 0, 255)
        elif i in range(76, 88):  # outer mouth
            COLOR = (255, 255, 148)
        elif i in range(88, 96):  # inner mouth
            COLOR = (0, 0, 255)
        elif i in range(96, 98):  # pupils
            COLOR = (0, 255, 0)
        cv2.circle(frame, (x, y), 0, COLOR, 2)
        if draw_index:
            cv2.putText(frame, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.2, (255, 255, 255), 1, cv2.LINE_AA)
    if draw_contour:
        left_eye = np.array(coords[60:68]).reshape((-1, 1, 2)).astype(np.int32)
        right_eye = np.array(coords[68:76]).reshape(
            (-1, 1, 2)).astype(np.int32)
        outer_mouth = np.array(coords[76:88]).reshape(
            (-1, 1, 2)).astype(np.int32)
        face_region = np.array(coords[0:33]).reshape(
            (-1, 1, 2)).astype(np.int32)
        cv2.drawContours(frame, [left_eye], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [right_eye], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [outer_mouth], -1, (0, 255, 0), 1)

    width = int((x2-x1)/2)
    height = int((y2-y1)/2-PADDING/2)
    # direction_vector = get_3d_direction(
    #     frame, (x1, y1), (width, height), x_points, y_points)
    # draw_bounding_cube(frame, coords, direction_vector)
    return frame


def show_face(face, key_points):
    feature_index = [60, 72, 54, 76, 82, 16]
    PADDING = 32
    height, width, _ = face.shape
    scale_x = width/input_shape
    scale_y = height/input_shape
    x_points = key_points[::2]
    y_points = key_points[1::2]
    x_points = np.array(x_points)*scale_x+PADDING/2
    x_points = x_points.astype(int)
    y_points = np.array(y_points)*scale_y+PADDING/2
    y_points = y_points.astype(int)
    coords = list(zip(x_points, y_points))
    frame = np.zeros((height+PADDING, width+PADDING, 3))
    for i, coord in enumerate(coords, start=0):
        x, y = coord
        COLOR = (255, 133, 71)  # face region
        if i in range(33, 51):  # eyebrows
            COLOR = (0, 255, 0)
        elif i in range(51, 55):  # nose1
            COLOR = (255, 255, 255)
        elif i in range(55, 60):  # nose2
            COLOR = (0, 255, 255)
        elif i in range(60, 76):  # eye bounds
            COLOR = (255, 0, 255)
        elif i in range(76, 88):  # outer mouth
            COLOR = (255, 255, 148)
        elif i in range(88, 96):  # inner mouth
            COLOR = (0, 0, 255)
        elif i in range(96, 98):  # pupils
            COLOR = (0, 255, 0)
        cv2.circle(frame, (x, y), 0, COLOR, 1)
        cv2.putText(frame, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.275, (255, 255, 255), 1, cv2.LINE_AA)
    get_3d_direction(frame, (0, 0), (width/2, height/2), x_points, y_points)
    frame = cv2.resize(frame, (210, 300), interpolation=cv2.INTER_AREA)
    # cv2.imshow("face frame", frame)
    return frame


def apply_sharpen(frame):
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    frame = cv2.filter2D(frame, -1, kernel)
    return frame


while True:
    try:
        _, frame = cap.read()
        HEIGHT, WIDTH, _ = frame.shape
        clahe_frame = apply_clahe(frame)  # use clahe to adjust brightness
        frame = clahe_frame.copy()
        gamma_boost_frame = adjust_gamma(
            clahe_frame, gamma=1.40)  # boost gamma
        sharper_frame = apply_sharpen(gamma_boost_frame)  # sharpen image
        blob = cv2.dnn.blobFromImage(clahe_frame, 1.0, (300, 300), [
            104, 117, 123], False, False)
        net.setInput(blob)
        faces = net.forward()
        for i in range(faces.shape[2]):
            if i > 0:  # grab only first and one face
                break
            confidence = faces[0, 0, i, 2]
            # print("Face confidence {}".format(confidence))
            if confidence > 0.7:
                face_found = True
                x1 = int(faces[0, 0, i, 3] * WIDTH)
                y1 = int(faces[0, 0, i, 4] * HEIGHT)
                x2 = int(faces[0, 0, i, 5] * WIDTH)
                y2 = int(faces[0, 0, i, 6] * HEIGHT)
                width = abs(x2-x1)
                height = abs(y2-y1)
                offset_x = 0
                offest_y = 0
                if height > width:
                    offset_x = (height - width)/2
                    offset_x = int(offset_x)
                else:
                    offest_y = (width - height)/2
                    offest_y = int(offest_y)+10
        face = sharper_frame[y1:y2, x1:x2]  # crop only face
        # convert color of face image
        grey_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        grey_face = cv2.resize(grey_face, (input_shape, input_shape),
                               interpolation=cv2.INTER_AREA)  # resize the image
        key_points = detect_points(grey_face)  # facial landmarks points
        draw_face_estimator(frame, key_points, draw_index=False,
                            draw_point=True, draw_contour=True)
        MAR = calculate_mar(key_points)
        L_EAR, R_EAR, AVG_EAR = calculate_ear(key_points)
        # print(MAR, L_EAR, R_EAR, AVG_EAR)
        overlay = show_face(face, key_points)
        o_height, o_width, _ = overlay.shape
        frame[0:o_height, WIDTH-o_width:WIDTH] = overlay

    except Exception as err:
        # print(err)
        pass
    if show_vdo:
        cv2.imshow("frame", frame)
        key = cv2.waitKey(1) & 0xff
        if ord("q") == key:
            break
cap.release()
cv2.destroyAllWindows()
