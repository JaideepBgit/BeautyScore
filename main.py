# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 1 11:48:50 2021

@author: Jaideep Bommidi
"""


#################################
#Import Essential Libraries
#################################
#import age_gender_detector
import dlib
from imutils import face_utils
import numpy as np
import imutils
import cv2
import math
from deepface import DeepFace
from utils import CvFpsCalc, utils, hairNet_matting, mediapipe_custom
from face_mesh.face_mesh import FaceMesh
from iris_landmark.iris_landmark import IrisLandmark
import copy
import time

#################################
#Define Most Useful Functions:
#################################

LEFT_EYE_INDICES = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_INDICES = [42, 43, 44, 45, 46, 47]
GOLDEN_VALUE = 1.61803398875
COEFFICIENT = 2.38196601125

#https://stackoverflow.com/questions/47246540/how-to-draw-a-line-from-two-points-and-then-let-the-line-complete-drawing-until

"""
Iris backgdrop
"""
face_meshIris = FaceMesh(
    1,
    0.7,
    0.7,
)
iris_detector = IrisLandmark()
cvFpsCalc = CvFpsCalc(buffer_len=10)
display_fps = cvFpsCalc.get()



def distance(pt1,pt2):
  return math.sqrt((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)


def angle_between_2_points(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    tan = (y2 - y1) / (x2 - x1)
    return np.degrees(np.arctan(tan))

def get_rotation_matrix(p1, p2):
    angle = angle_between_2_points(p1, p2)
    x1, y1 = p1
    x2, y2 = p2
    xc = (x1 + x2) // 2
    yc = (y1 + y2) // 2
    M = cv2.getRotationMatrix2D((xc, yc), angle, 1)
    return M

def extract_eye_center(shape, eye_indices):
    points = extract_eye(shape, eye_indices)
    xs = map(lambda p: p.x, points)
    ys = map(lambda p: p.y, points)
    return sum(xs) // 6, sum(ys) // 6

def extract_eye(shape, eye_indices):
    points = map(lambda i: shape.part(i), eye_indices)
    return list(points)

def extract_left_eye_center(shape):
    return extract_eye_center(shape, LEFT_EYE_INDICES)

def extract_right_eye_center(shape):
    return extract_eye_center(shape, RIGHT_EYE_INDICES)

#################################
#Define Most Useful Functions:
#################################


golden_ratio_v = list()
golden_ratio_h = list()
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./models/shape_predictor_68_face_landmarks.dat")


#################################
#Load the input image, resize it, and convert it to grayscale
#################################
import os
import configparser
#initialize the config file
config = configparser.ConfigParser(allow_no_value=True)
#read the config file
config.read_file(open("./configuration.ini"))

from os import listdir
from os.path import isfile, join
mypath = config["Folders"]["imagesFolder"]
IMAGE_FILES = [f for f in listdir(mypath) if isfile(join(mypath, f))]

#################################
# Numbering of the vertices
#################################
cnt=0
# font
font = cv2.FONT_HERSHEY_SIMPLEX

# fontScale
fontScale = 0.25
   
# Blue color in BGR
color = (255, 0, 0)
  
# Line thickness of 2 px
thickness = 1

for image_file_name in IMAGE_FILES:
    print("###########################################################################")
    print(image_file_name)
    print("---------------------------------------------------------------------------")
    loc = os.path.join(mypath,image_file_name)
    image = cv2.imread(loc)

    """
    Apply hair matting
    """
    mask = hairNet_matting.predict(image)
    st = time.time()
    d1 = time.time()
    dst = hairNet_matting.transfer(image, mask)
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    #cv2.imwrite(os.path.join(config["Folders"]["resultsFolder"],image_file_name.strip(".jpg")+"_mask.jpg"),dst)
    """
    #facemesh call : MEDIAPIPE
    """
    image, silhouette_image, idx_to_coordinates = mediapipe_custom.mediapipe_func(image)

    #draw face mesh points for understanding, mediapipe
    #[points on face]
    """
    Uncomment to draw POINTS
    """
    #image = utils.draw_mediapipe_landmarks(image,idx_to_coordinates)
    """
    # Draw traingles and shapes
    """
    #image = utils.draw_shapes(image,idx_to_coordinates)
    
    """
    # calculate the eyebrow and lip width
    """
    right_eyebrow, left_eyebrow, Lower_lip, Upper_lip = utils.cal_eyebrow_height_width(idx_to_coordinates)
    """
    """
    #ALL ABOUT FACE
    attributes = ['age', 'gender' , 'race' , 'emotion']
    try:
        analysis = DeepFace.analyze(loc,attributes)
    except ValueError as Errorval:
        print("Face could not be detected")
        continue
    
    print(analysis['age'],analysis['gender'],analysis['dominant_race'],analysis['dominant_emotion'])
    
    #RESIZE AND CONVERT TO GRAYSCALE IMAGE
    #image = imutils.resize(image, width=512)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # detect faces in the grayscale image
    rects = detector(gray, 1)
    shape=[]
    # loop over the face detections
    for (i, rect) in enumerate(rects):
    	# determine the facial landmarks for the face region, then
    	# convert the facial landmark (x, y)-coordinates to a NumPy
    	# array
        shape = predictor(gray, rect)
        # rotate to good vision
        left_eye = extract_left_eye_center(shape)
        right_eye = extract_right_eye_center(shape)
        M = get_rotation_matrix(left_eye, right_eye)
        image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC)
        gray = cv2.warpAffine(gray, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC)
    #################################
    # detect faces in the grayscale image
    rects = detector(gray, 1)
    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
    	# convert the facial landmark (x, y)-coordinates to a NumPy
    	# array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        # convert dlib's rectangle to a OpenCV-style bounding box
    	# [i.e., (x, y, w, h)], then draw the face bounding box
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        #def detected_face_to_file(image,x,y,w,h,config,image_file_name):
        image_copy = utils.create_blank_image(image.shape)
        image_copy = copy.deepcopy(image)
        utils.detected_face_to_file(image_copy,x,y,w,h,config,image_file_name)

        
    """
    Draw 68 landmarks on face (from Dlib)
    """
    #utils.draw_mediapipe_landmarks(image, shape,"dlib")
    
    """
    Calculate Golden Ratio
    """
    golden_ratio_h, golden_ratio_v, center_of_left_pupil, center_of_right_pupil,nose_at_nostrils_right, nose_at_nostrils_left = utils.cal_golden_ratio(golden_ratio_v,golden_ratio_h,image,shape)
    
    """
    # Draw lines
    """
    params =[idx_to_coordinates,shape,center_of_left_pupil,center_of_right_pupil,nose_at_nostrils_right,nose_at_nostrils_left,mask] 
    image,hair_face_point = utils.draw_Lines(image,params,False)
    
    """
    Calculate shapes of face
    """
    shape_type_obj = utils.find_face_shape_type(hair_face_point, shape[8],idx_to_coordinates[54],idx_to_coordinates[284],idx_to_coordinates[227],idx_to_coordinates[447],idx_to_coordinates[138],idx_to_coordinates[367])
    shape_type_obj.dist_1_calc()
    shape_type_obj.dist_2_calc()
    shape_type_obj.dist_3_calc()
    shape_type_obj.dist_4_calc()
    face_shape = shape_type_obj.find_face_shape_type_meth()
    
    #print((np.mean(golden_ratio_v)+np.mean(golden_ratio_h))/2)
    golden_ratio = golden_ratio_v+golden_ratio_h
    print("BEAUTY:",np.mean([(1 - abs(g_vh - GOLDEN_VALUE)/COEFFICIENT)*100 for g_vh in golden_ratio]))
    dst = hairNet_matting.transfer(image, mask)
    
    """
    Detect wrinkles
    """
    tempImg = utils.create_blank_image(image.shape)# Extract out the object and place into output image
    tempImg = copy.deepcopy(image)
    facecountourPoints = [10,109,67,103,54,21,127,227,137,177,215,172,136,150,149,176,148,152,377,400,378,379,365,397,288,435,361,401,323,454,264,356,389,251,284,332,297,338,10]
    #tempImg = utils.EdgedrawOutline(tempImg, facecountourPoints)
    #cv2.imshow("tempImg",tempImg)
    #cv2.waitKey(0)
    #tempImg[mask == 255] = image[mask == 255]
    
    w,h,_ = dst.shape
   #dst =  img[y:y+h, x:x+w]
    dst = dst[int(0.1*w):int(0.9*w),int(0.1*h):int(0.9*h)]
    w,h,_ = image.shape
    image = image[int(0.1*w):int(0.9*w),int(0.1*h):int(0.9*h)]
    w,h,_ = silhouette_image.shape
    silhouette_image = silhouette_image[int(0.1*w):int(0.9*w),int(0.1*h):int(0.9*h)]
    if True:
        cv2.putText(image, face_shape, (50,100), font, 1, (0,0,255), 3, cv2.LINE_AA)
    cv2.putText(silhouette_image, face_shape, (50,100), font, 1, (0,0,255), 3, cv2.LINE_AA)
    cv2.imwrite(os.path.join(config["Folders"]["HairNet"],image_file_name.strip(".jpg")+"_mask.jpg"),dst)
    cv2.imwrite(os.path.join(config["Folders"]["resultsFolder"],image_file_name),image)
    cv2.imwrite(os.path.join(config["Folders"]["silhouette"],image_file_name.strip(".jpg")+"_silhouette.jpg"),silhouette_image)
    cv2.imshow("Image",image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()