# -*- coding: utf-8 -*-
"""
Created on Fri Dec 2 12:13:11 2021

@author: Jaideep Bommidi
"""
from face_mesh.face_mesh import FaceMesh
import mediapipe as mp
import cv2
from utils import utils
"""
#working with google ML
"""

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
#face mesh
mp_face_mesh = mp.solutions.face_mesh
#face mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
mp_drawing_styles = mp.solutions.drawing_styles


def collect_points(face_results):
    face_points = []
    try:
        for i in face_results[0]:
            face_points.append((i[0],i[1]))
    except IndexError:
        pass
    return face_points
def mediapipe_func(image):
    """
    Google mediapipe
    """
    with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5) as face_detection, mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:
        results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        results_faceMesh = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        try:
            for detection in results.detections:
              print('Nose tip:')
    
              xy  = mp_face_detection.get_key_point(
                  detection, mp_face_detection.FaceKeyPoint.NOSE_TIP)
              print(mp_face_detection.get_key_point(
                  detection, mp_face_detection.FaceKeyPoint.NOSE_TIP))
              imgX,imgY,imgZ = image.shape
              Nosepoint = (imgX*xy.x,imgY*xy.y)
              #keypoint_px = mp_drawing.draw_detection((0,0),annotated_image, detection)
              #print(keypoint_px)
              if not results_faceMesh.multi_face_landmarks:
                  continue
            for face_landmarks in results_faceMesh.multi_face_landmarks:
                image,idx_to_coordinates = mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_contours_style())
            lower_lip_curve_locs = [78,95,88,178,87,14,317,402,318,324,308]
            lower_lip_curve_locs_pairs = []
            outter_lip_points = [0,267,269,270,409,306,375,321,405,314,17,84,181,91,146,61,185,40,39,37,0]
            inner_lip_points = [13,312,311,310,415,308,324,318,402,317,14,87,178,88,95,78,191,80,81,82,13]
            #[0,13,14,17,37,39,40,61,78,80,81,82,84,87,88,91,95,146,178,185,191,267,269,270,292,306,308,310,311,312,317,318,324,375,402,405,409,415]
            face_contour = [10,109,67,103,54,21,127,227,137,177,215,172,136,150,149,176,148,152,377,400,378,379,365,397,288,435,361,401,323,454,264,356,389,251,284,332,297,338,10]
            right_eyebrow = [70,46,53,52,65,55,107,66,105,63,70]
            left_eyebrow = [336,285,295,282,283,276,300,293,334,296,336]
            silhouette_image = utils.create_blank_image(image.shape)
            facepoints_obj = utils.facepoints(image.shape, face_contour, outter_lip_points,inner_lip_points, left_eyebrow, right_eyebrow)
            silhouette_image = facepoints_obj.draw(silhouette_image,idx_to_coordinates)
            for cnt, i in enumerate(lower_lip_curve_locs):
                if cnt < len(lower_lip_curve_locs)-1:
                    lower_lip_curve_locs_pairs.append((i,lower_lip_curve_locs[cnt+1]))
        except TypeError as Error:
            print("There is no Face")
    return image, silhouette_image, idx_to_coordinates