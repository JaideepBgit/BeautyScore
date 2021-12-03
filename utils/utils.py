# -*- coding: utf-8 -*-
"""
Created on Wed Dec 1 10:29:04 2021

@author: Jaideep Bommidi
"""
import numpy as np
import cv2
import math
import os
#################################
# NUmbering of the vertices
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
def distance(pt1,pt2):
  return math.sqrt((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)

def create_blank_image(shape):
    return np.zeros(shape,np.uint8)

def detected_face_to_file(image,x,y,w,h,config,image_file_name):
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imwrite(os.path.join(config["Folders"]["DetectedFace"],image_file_name.strip(".jpg")+"_face.jpg"),image)

class facepoints:
    def __init__(self,shape,facecontour, outter_lip,inner_lip, lefteyebrows,righteyebrows):
        self.shape = shape
        self.facecontour = facecontour
        self.outter_lip = outter_lip
        self.inner_lip = inner_lip
        self.lefteyebrows = lefteyebrows
        self.righteyebrows = righteyebrows
    def draw(self,image, points):
        for cntf,i in enumerate(self.facecontour):
            if cntf < len(self.facecontour)-1:
                cv2.line(image, points[i], points[self.facecontour[cntf+1]], (255, 255, 255), thickness=1)
                #cv2.putText(image, str(i), points[i], font, 0.4, (0,0,255), 1, cv2.LINE_AA)
        for cntl,i in enumerate(self.outter_lip):
            if cntl < len(self.outter_lip)-1:
                cv2.line(image, points[i], points[self.outter_lip[cntl+1]], (255, 255, 255), thickness=1)
        for cntl,i in enumerate(self.inner_lip):
            if cntl < len(self.inner_lip)-1:
                cv2.line(image, points[i], points[self.inner_lip[cntl+1]], (255, 255, 255), thickness=1)
        for cntle,i in enumerate(self.lefteyebrows):
            if cntle < len(self.lefteyebrows)-1:
                cv2.line(image, points[i], points[self.lefteyebrows[cntle+1]], (255, 255, 255), thickness=1)
        for cntre,i in enumerate(self.righteyebrows):
            if cntre < len(self.righteyebrows)-1:
                cv2.line(image, points[i], points[self.righteyebrows[cntre+1]], (255, 255, 255), thickness=1)
        return image
def EdgedrawOutline(image,points):
    cv2.drawContours(image, points, -1, 255, 3) # Draw filled contour in mask
class find_face_shape_type:
    def __init__(self,hair_line_pt, chin_line_sp,p21,p22,p31,p32,p41,p42):
        self.hair_line_pt = hair_line_pt
        self.chin_line_sp = chin_line_sp
        self.shape = None #["Oval","Oblong","Square","Heart","Diamond"]
        self.dist_1= 0   
        self.p21 = p21
        self.p22 = p22
        self.dist_2 = 0
        self.p31 = p31
        self.p32 = p32
        self.dist_3 = 0
        self.p41 = p41
        self.p42 = p42
        self.dist_4 = 0
    def dist_1_calc(self):
        dist_1_real  = distance(self.hair_line_pt,self.chin_line_sp)
        dist_1_straight_not_real = distance(self.hair_line_pt,(self.hair_line_pt[0],self.chin_line_sp[1]))
        self.dist_1 = int((dist_1_real+dist_1_straight_not_real)/2)
        print("distance 1: {}".format(self.dist_1))
    def dist_2_calc(self):
        self.dist_2  = int(distance(self.p21,self.p22))
        print("distance 2: {}".format(self.dist_2))
    def dist_3_calc(self):
        self.dist_3 = int(distance(self.p31,self.p32))
        print("distance 3: {}".format(self.dist_3))
    def dist_4_calc(self):
        self.dist_4 = int(distance(self.p41,self.p42))
        print("distance 4: {}".format(self.dist_4))
    def find_face_shape_type_meth(self):
        if self.dist_1 > self.dist_2 and self.dist_1>self.dist_3 and self.dist_1>self.dist_4:
            if self.dist_3-100 > self.dist_2 and self.dist_3-100 > self.dist_4:
                print("Diamond shape")
                return "Diamond Shape"
            if self.dist_1/2 > self.dist_3:
                print("Oval shape")
                return "Oval Shape"
            if self.dist_3 in range(self.dist_2-100,self.dist_2+100) and self.dist_4 in range(self.dist_3-120,self.dist_3+100) and self.dist_2 in range(self.dist_4-50,self.dist_4+50):
                print("Square shape")
                return "Square shape"
            if self.dist_4 <self.dist_2 and self.dist_4<self.dist_3:
                if self.dist_2 in range(self.dist_3-10,self.dist_3+10):
                    print("Heart shape")
                    return "Heart Shape"
        else:
            print("Shortest face")
            return "Shortest face"
        return "Round"


def draw_mediapipe_landmarks(image, idx_to_coordinates,name = "mediapipe"):
    if name=="mediapipe":
        for cntf,i in enumerate(idx_to_coordinates):
            #if cnt in [46,52,53,55,65,276,282,283,285,295]:
            if cntf == 460:
                x=1
                pass
            if cntf in [11,12,15,16,42,41,38,74,73,72,61,62,76,77,78,80,81,82,86,85,87,88,89,90,91,96,195,146,178,179,180,183,184,191,267,268,269,270,271,272,291,292,302,303,304,306,307,308,310,311,312,316,315,317,318,319,320,324,325,403,404,408,407]:
                if cntf not in [0,61,78,80,81,82,84,87,88,91,146,178,185,191,267,269,270,306,308,310,311,312,317,318,324,375,402,405,409,415]:
                    continue
            if cntf in [10,109,67,103,54,21,62,127,227,137,177,215,172,136,150,149,176,148,152,377,400,378,379,365,397,288,435,361,401,366,323,447,454,264,356,368,389,251,284,332,297,338]:
                cv2.putText(image, str(cntf), idx_to_coordinates[i], font, 0.4, (0,0,255), 1, cv2.LINE_AA)
            else:
                cv2.putText(image, str(cntf), idx_to_coordinates[i], font, 0.4, (0,0,0), 1, cv2.LINE_AA)
            #pass
    elif name=="dlib":
        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        cnt = 1
        for (x, y) in idx_to_coordinates:
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
            image = cv2.putText(image, str(cnt), (x, y), font,fontScale, color, thickness, cv2.LINE_AA)
            cnt +=1
    return image

def draw_shapes(image,idx_to_coordinates):
    """
    Right face triangles
    """
    #[111,128,203]
    
    #[100,111,187]:
    # connect points for triangles on face
    cv2.line(image, idx_to_coordinates[111], idx_to_coordinates[128], (255, 255, 255), thickness=1)
    cv2.line(image, idx_to_coordinates[128], idx_to_coordinates[203], (255, 255, 255), thickness=1)
    cv2.line(image, idx_to_coordinates[203], idx_to_coordinates[111], (255, 255, 255), thickness=1)
    # connect points for triangles on face
    cv2.line(image, idx_to_coordinates[116], idx_to_coordinates[92], (0, 0, 0), thickness=1)
    cv2.line(image, idx_to_coordinates[92], idx_to_coordinates[215], (0, 0, 0), thickness=1)
    cv2.line(image, idx_to_coordinates[215], idx_to_coordinates[116], (0, 0, 0), thickness=1)
    """
    Left face triangles
    """
    #[346,412,278]:
    # connect points for triangles on face, TOP triangle
    cv2.line(image, idx_to_coordinates[346], idx_to_coordinates[412], (255, 255, 255), thickness=1)
    cv2.line(image, idx_to_coordinates[412], idx_to_coordinates[278], (255, 255, 255), thickness=1)
    cv2.line(image, idx_to_coordinates[278], idx_to_coordinates[346], (255, 255, 255), thickness=1)
    
    #[329,346,427]:
    # connect points for triangles on face, BOTTOM triangle
    cv2.line(image, idx_to_coordinates[322], idx_to_coordinates[345], (0, 0, 0), thickness=1)
    cv2.line(image, idx_to_coordinates[345], idx_to_coordinates[435], (0, 0, 0), thickness=1)
    cv2.line(image, idx_to_coordinates[435], idx_to_coordinates[322], (0, 0, 0), thickness=1)
    """
    CHIN Triangles
    """
    #[83,313,175]
    # connect points for triangles on face
    cv2.line(image, idx_to_coordinates[83], idx_to_coordinates[313], (255, 255, 255), thickness=1)
    cv2.line(image, idx_to_coordinates[313], idx_to_coordinates[175], (255, 255, 255), thickness=1)
    cv2.line(image, idx_to_coordinates[175], idx_to_coordinates[83], (255, 255, 255), thickness=1)
    
    """
    Forehead triangles
    middle: [69,299,9/168]
    """
    cv2.line(image, idx_to_coordinates[69], idx_to_coordinates[299], (255, 255, 255), thickness=1)
    cv2.line(image, idx_to_coordinates[299], idx_to_coordinates[9], (255, 255, 255), thickness=1)
    cv2.line(image, idx_to_coordinates[9], idx_to_coordinates[69], (255, 255, 255), thickness=1)
    
    #left
    cv2.line(image, idx_to_coordinates[109], idx_to_coordinates[54], (0, 0, 0), thickness=1)
    cv2.line(image, idx_to_coordinates[54], (idx_to_coordinates[105][0]-2,idx_to_coordinates[105][1]-2), (0, 0, 0), thickness=1)
    cv2.line(image, (idx_to_coordinates[105][0]-2,idx_to_coordinates[105][1]-2), idx_to_coordinates[109], (0, 0, 0), thickness=1)
    #right
    cv2.line(image, idx_to_coordinates[338], idx_to_coordinates[284], (0, 0, 0), thickness=1)
    cv2.line(image, idx_to_coordinates[284], (idx_to_coordinates[334][0]-2,idx_to_coordinates[334][1]-2), (0, 0, 0), thickness=1)
    cv2.line(image, (idx_to_coordinates[334][0]-2,idx_to_coordinates[334][1]-2), idx_to_coordinates[338], (0, 0, 0), thickness=1)
    
    """
    Eyebrow contour
    """
    #connect missing eyebrows to make a contour
    #right
    cv2.line(image, idx_to_coordinates[70], idx_to_coordinates[46], (0, 0, 255), thickness=2)
    cv2.line(image, idx_to_coordinates[107], idx_to_coordinates[55], (0, 0, 255), thickness=2)
    #left
    cv2.line(image, idx_to_coordinates[336], idx_to_coordinates[285], (0, 255, 0), thickness=2)
    cv2.line(image, idx_to_coordinates[276], idx_to_coordinates[300], (0, 255, 0), thickness=2)
    return image
    
def cal_eyebrow_height_width(idx_to_coordinates):
    """
    Eyebrow width and height
    """
    #right
    dist_70_46 = distance(idx_to_coordinates[70],idx_to_coordinates[46])
    dist_63_53 = distance(idx_to_coordinates[63],idx_to_coordinates[53])
    dist_105_52 = distance(idx_to_coordinates[105],idx_to_coordinates[52])
    dist_66_65 = distance(idx_to_coordinates[66],idx_to_coordinates[65])
    dist_107_55 = distance(idx_to_coordinates[107],idx_to_coordinates[55])
    right_eyebrow = [dist_70_46,dist_63_53,dist_105_52,dist_66_65,dist_107_55]
    right_eyebrow = [round(val_conv,2) for val_conv in right_eyebrow]
    print("Right Eyebrow: {:.2f},{:.2f},{:.2f},{:.2f},{:.2f}".format(dist_107_55,dist_66_65,dist_105_52,dist_63_53,dist_70_46))
    #left
    dist_336_285 = distance(idx_to_coordinates[336],idx_to_coordinates[285])
    dist_296_295 = distance(idx_to_coordinates[296],idx_to_coordinates[295])
    dist_334_282 = distance(idx_to_coordinates[334],idx_to_coordinates[282])
    dist_293_283 = distance(idx_to_coordinates[293],idx_to_coordinates[283])
    dist_300_276 = distance(idx_to_coordinates[300],idx_to_coordinates[276])
    left_eyebrow = [dist_336_285,dist_296_295,dist_334_282,dist_293_283,dist_300_276]
    left_eyebrow = [round(val_conv,2) for val_conv in left_eyebrow]
    print("Left Eyebrow: {:.2f},{:.2f},{:.2f},{:.2f},{:.2f}".format(dist_336_285,dist_296_295,dist_334_282,dist_293_283,dist_300_276))
    
    """
    Lips width and height
    """
    #lower lip
    dist_14_17 = distance(idx_to_coordinates[14],idx_to_coordinates[17])
    dist_317_314 = distance(idx_to_coordinates[317],idx_to_coordinates[314])
    dist_402_405 = distance(idx_to_coordinates[402],idx_to_coordinates[405])
    dist_318_321 = distance(idx_to_coordinates[318],idx_to_coordinates[321])
    dist_88_91 = distance(idx_to_coordinates[88],idx_to_coordinates[91])
    dist_178_181 = distance(idx_to_coordinates[178],idx_to_coordinates[181])
    dist_87_84 = distance(idx_to_coordinates[87],idx_to_coordinates[84])
    Lower_lip = [dist_14_17,dist_317_314,dist_402_405,dist_318_321,dist_88_91,dist_178_181,dist_87_84]
    Lower_lip = [round(val_conv,2) for val_conv in Lower_lip]
    print("Lower lip: {:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}".format(dist_14_17,dist_317_314,dist_402_405,dist_318_321,dist_88_91,dist_178_181,dist_87_84))
    #upper lip
    dist_0_13 = distance(idx_to_coordinates[0],idx_to_coordinates[13])
    dist_267_312 = distance(idx_to_coordinates[267],idx_to_coordinates[312])
    dist_269_311 = distance(idx_to_coordinates[269],idx_to_coordinates[311])
    dist_37_82 = distance(idx_to_coordinates[37],idx_to_coordinates[82])
    dist_39_81 = distance(idx_to_coordinates[39],idx_to_coordinates[81])
    dist_40_80 = distance(idx_to_coordinates[40],idx_to_coordinates[80])
    Upper_lip = [dist_0_13,dist_267_312,dist_269_311,dist_37_82,dist_39_81,dist_40_80]
    Upper_lip = [round(val_conv,2) for val_conv in Upper_lip]
    print("Upper Lip: {:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}".format(dist_0_13,dist_267_312,dist_269_311,dist_37_82,dist_39_81,dist_40_80))

    return right_eyebrow, left_eyebrow, Lower_lip, Upper_lip

def cal_golden_ratio(golden_ratio_v,golden_ratio_h,image,shape):
    #################################
    #V1: Center of pupils ,Center of lips, Bottom of chin
    #################################
    try:
        center_of_left_pupil = ((shape[43]+shape[44])[0]//2 , (shape[43]+shape[47])[1]//2)
    except IndexError:
        return "No face detected"
    cnt=1
    golden_ratio_v = list()
    golden_ratio_h = list()
    cv2.circle(image, center_of_left_pupil, 4, (255, 0, 255), -1)
    image = cv2.putText(image, str(cnt), center_of_left_pupil, font, 
                       fontScale, color, thickness, cv2.LINE_AA)
    cnt +=1
    center_of_right_pupil = ((shape[37]+shape[38])[0]//2 , (shape[37]+shape[41])[1]//2)
    cv2.circle(image,center_of_right_pupil, 4, (255, 0, 255), -1)
    image = cv2.putText(image, str(cnt), center_of_right_pupil, font, 
                       fontScale, color, thickness, cv2.LINE_AA)
    cnt +=1
    (x_mouth_center,y_mouth_center) = (shape[63]+shape[67])//2
    mouth_center = (x_mouth_center , y_mouth_center)
    cv2.circle(image, (x_mouth_center,y_mouth_center), 2, (255,255,0), -1)
    image = cv2.putText(image, str(cnt), (x_mouth_center,y_mouth_center), font, 
                       fontScale, color, thickness, cv2.LINE_AA)
    cnt +=1
    right_pupil_to_center_of_lips = center_of_right_pupil[1] - mouth_center[1]
    #print(right_pupil_to_center_of_lips)
    left_pupil_to_center_of_lips = center_of_left_pupil[1] - mouth_center[1]
    center_of_lips_to_chin = mouth_center[1] - shape[8][1]
    
    V1 = abs(right_pupil_to_center_of_lips/center_of_lips_to_chin)
    golden_ratio_v.append( V1 )
    
    
    #################################
    #V2: Center of pupils ,Nose at nostrils, Bottom of chin
    #################################
    (nose_at_nostrils_x_right,_) = (shape[35]+shape[42])//2
    nose_at_nostrils_right = (nose_at_nostrils_x_right,shape[35][1])
    (nose_at_nostrils_x_left,_) = (shape[31]+shape[39])//2
    nose_at_nostrils_left = (nose_at_nostrils_x_left,shape[31][1])
    #cv2.circle(image, nose_at_nostrils_right, 2, (255,255,255) , -1)
    #image = cv2.putText(image, str(cnt), nose_at_nostrils_right, font, 
    #                   fontScale, color, thickness, cv2.LINE_AA)
    cnt +=1
    #cv2.circle(image, nose_at_nostrils_left, 2, (255,255,255) , -1)
    #image = cv2.putText(image, str(cnt), nose_at_nostrils_left, font, 
    #                   fontScale, color, thickness, cv2.LINE_AA)
    cnt +=1
    right_nose_nostrils_to_right_pupil = nose_at_nostrils_right[1] - center_of_right_pupil[1]
    left_nose_nostrils_to_left_pupil = nose_at_nostrils_left[1] - center_of_right_pupil[1]
    average_of_nostrilses_to_pupils = (right_nose_nostrils_to_right_pupil+left_nose_nostrils_to_left_pupil)/2
    
    chin = shape[8]
    
    right_nose_nostrils_to_chin = nose_at_nostrils_right[1] - chin[1]
    left_nose_nostrils_to_chin = nose_at_nostrils_left[1] - chin[1]
    average_of_nostrilses_to_chin = (right_nose_nostrils_to_chin+left_nose_nostrils_to_chin)/2
    
    #TEST
    #cv2.line(image,(center_of_right_pupil[0],center_of_right_pupil[1]),(center_of_right_pupil[0],nose_at_nostrils_right[1]),(255,120,255),2)
    #cv2.line(image,(nose_at_nostrils_right[0],nose_at_nostrils_right[1]),(nose_at_nostrils_right[0],chin[1]),(255,120,255),2)
    V2 =  abs(average_of_nostrilses_to_chin/average_of_nostrilses_to_pupils)
    golden_ratio_v.append(V2)
    
    
    #################################
    #V3: Center of pupils ,Nose flair top , Nose base
    #################################
    
    nose_flair_top_left = ((shape[29]+shape[39])[0]//2 , (shape[29][1]+shape[30][1])//2)
    nose_flair_top_right = ((shape[29]+shape[42])[0]//2 , (shape[29][1]+shape[30][1])//2)
    
    
    #cv2.circle(image, nose_flair_top_left, 2, (255,12,98) , -1)
    #image = cv2.putText(image, str(cnt), nose_flair_top_left, font, 
    #                   fontScale, color, thickness, cv2.LINE_AA)
    cnt +=1
    #cv2.circle(image, nose_flair_top_right, 2, (255,12,98) , -1)
    #image = cv2.putText(image, str(cnt), nose_flair_top_right, font, 
    #                   fontScale, color, thickness, cv2.LINE_AA)
    cnt +=1
    
    pupil_to_flair_top = center_of_left_pupil[1] - nose_flair_top_left[1]
    
    
    left_nose_base = shape[33][1]
    
    flair_to_nose_base = nose_flair_top_left[1] - left_nose_base
    
    
    V3 = abs(pupil_to_flair_top/flair_to_nose_base)
    golden_ratio_v.append(V3)
    
    
    
    #####################################################
    #V4: Top arc of eyebrows ,Top of eyes ,Bottom of eyes
    #####################################################
    top_arc_of_eyebrows = shape[19]
    top_of_eyes = shape[37]
    bottom_of_eyes = shape[41]
    
    #TEST
    #cv2.line(image,(top_arc_of_eyebrows[0],top_arc_of_eyebrows[1]),(top_arc_of_eyebrows[0],top_of_eyes[1]),(255,120,255),2)
    #cv2.line(image,(top_of_eyes[0],top_of_eyes[1]),(top_of_eyes[0],bottom_of_eyes[1]),(255,120,255),2)
    
    V4 = abs((top_arc_of_eyebrows[1] - top_of_eyes[1]) /(top_of_eyes[1] - bottom_of_eyes[1]))
    
    golden_ratio_v.append(V4)
    
    #######################################################
    #V5: Center of pupils ,Nose at nostrils ,Center of lips
    #######################################################
    V5 = abs(average_of_nostrilses_to_pupils / (nose_at_nostrils_right[1] - mouth_center[1]))
    
    golden_ratio_v.append(V5)
    
    
    #######################################################
    #V6: Top of lips ,Center of lips ,Bottom of lips ,
    #######################################################
    
    top_of_lips = shape[50]
    bottom_of_lips = shape[57]
    
    #TEST
    #cv2.line(image,(top_of_lips[0],top_of_lips[1]),(top_of_lips[0],mouth_center[1]),(255,120,255),2)
    #cv2.line(image,(mouth_center[0],mouth_center[1]),(mouth_center[0],bottom_of_lips[1]),(255,120,255),2)
    
    V6 = abs((mouth_center[1] - bottom_of_lips[1]) / (top_of_lips[1] - mouth_center[1]))
    
    golden_ratio_v.append(V6)
    
    #######################################################
    #V7: Nose at nostrils ,	Top of lips,Center of lips
    #######################################################
    
    V7 = abs( (nose_at_nostrils_left[1] - top_of_lips[1])/(top_of_lips[1] - mouth_center[1]))
    
    golden_ratio_v.append(V7)
    
    
    #######################################################
    #H1: Side of face ,	Inside of near eye , Opposite side of face
    #######################################################
    side_of_face = shape[0]
    inside_of_near_eye = shape[39]
    opposite_side_of_face = shape[16]
    
    #TEST
    #cv2.line(image,(side_of_face[0],side_of_face[1]),(inside_of_near_eye[0],side_of_face[1]),(255,120,255),2)
    #cv2.line(image,(inside_of_near_eye[0],inside_of_near_eye[1]),(opposite_side_of_face[0],inside_of_near_eye[1]),(255,255,120),2)
    
    H1 = abs((inside_of_near_eye[0] - opposite_side_of_face[0]) / (side_of_face[0] - inside_of_near_eye[0]))
    
    golden_ratio_h.append(H1)
    
    #######################################################
    #H2: Side of face 	Inside of near eye  Inside of opposite eye (16)
    #######################################################
    inside_of_opposite_eye = shape[42]
    H2 = abs( (side_of_face[0] - inside_of_near_eye[0])/(inside_of_near_eye[0] - inside_of_opposite_eye[0]))
    golden_ratio_h.append(H2)
    
    
    #######################################################
    #H3: Center of face , Outside edge of eye ,	Side of face
    #######################################################
    center_of_face = shape[8]
    outside_edge_of_eye = shape[36]
    H3 = abs( (center_of_face[0] - outside_edge_of_eye[0])/(outside_edge_of_eye[0] - side_of_face[0]))
    golden_ratio_h.append(H3)
    
    
    #######################################################
    #H4: Side of face 	Outside edge of eye	Inside edge of eye
    #######################################################
    inside_edge_of_eye = shape[39]
    
    #TEST
    #cv2.line(image,(side_of_face[0],side_of_face[1]),(outside_edge_of_eye[0],side_of_face[1]),(255,120,255),2)
    #cv2.line(image,(outside_edge_of_eye[0],outside_edge_of_eye[1]),(inside_edge_of_eye[0],outside_edge_of_eye[1]),(255,255,120),2)
    H4 = abs((side_of_face[0] - outside_edge_of_eye[0])/(outside_edge_of_eye[0] - inside_edge_of_eye[0]))
    golden_ratio_h.append(H4)
    
    
    
    #######################################################
    #H5: Side of face ,Outside of eye brow ,Outside edge of eye
    #######################################################
    outside_of_eye_brow = shape[17]
    
    #TEST
    #cv2.line(image,(side_of_face[0],side_of_face[1]),(outside_of_eye_brow[0],side_of_face[1]),(255,120,255),2)
    #cv2.line(image,(outside_of_eye_brow[0],outside_of_eye_brow[1]),(outside_edge_of_eye[0],outside_of_eye_brow[1]),(255,255,120),2)
    
    
    H5 = abs((outside_of_eye_brow[0] - outside_edge_of_eye[0])/(side_of_face[0] - outside_of_eye_brow[0]))
    golden_ratio_h.append(H5)
    
    
    
    #######################################################
    #H6: Center of face 	Width of nose 	Width of mouth
    #######################################################
    width_of_nose = nose_at_nostrils_right[0]
    width_of_mouth= shape[54][0]
    #cv2.line(image,(center_of_face[0],center_of_face[1]),(nose_at_nostrils_right[0],nose_at_nostrils_right[1]),(255,255,255))
    H6 = abs((center_of_face[0] - width_of_nose)/(width_of_nose - width_of_mouth))
    golden_ratio_h.append(H6)
    
    #######################################################
    #H7:Side of mouth ,Cupid’s bow ,Opposite side of mouth (30)
    #######################################################
    side_of_mouth = shape[48]
    cupids_bow = shape[52]
    opposite_side_of_mouth = shape[54]
    H7 = abs((side_of_mouth[0] - cupids_bow[0])/(cupids_bow[0] - opposite_side_of_mouth[0]))
    golden_ratio_h.append(H7)
    
    return golden_ratio_h, golden_ratio_v, center_of_left_pupil, center_of_right_pupil,nose_at_nostrils_right, nose_at_nostrils_left
def draw_extended_line(image,p1,p2):
    w,h,_ = image.shape
    theta = np.arctan2(p1[1]-p2[1], p1[0]-p2[0])
    """
    2000 this number should be generated/changed based on w and h
    """
    endpt1_x = int(p1[0] + 2000*np.cos(theta))
    endpt1_y = int(p1[1] + 2000*np.sin(theta))
    
    endpt2_x = int(p1[0] - 2000*np.cos(theta)/2)
    endpt2_y = int(p1[1] - 2000*np.sin(theta)/2)
    
    cv2.line(image, (endpt1_x, endpt1_y), (endpt2_x, endpt2_y), (0,0,0), 2)
    return image
def draw_Lines(image,params,putTextFlag):
    idx_to_coordinates = params[0]
    shape = params[1]
    center_of_left_pupil = params[2]
    center_of_right_pupil = params[3] 
    nose_at_nostrils_right = params[4]
    nose_at_nostrils_left = params[5]
    mask = params[6]
    #using mediapipe
    #image = draw_extended_line(image,idx_to_coordinates[55],idx_to_coordinates[285])
    image = draw_extended_line(image,idx_to_coordinates[55],idx_to_coordinates[55])
    if putTextFlag:
        cv2.putText(image, "Eyebrow bottom line", (idx_to_coordinates[55][0] - 10, idx_to_coordinates[55][1] - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    #eyebrow line top
    array_Y_eyebrow = [shape[18][1],shape[19][1],shape[24][1],shape[25][1]]
    ind_arr_Y_eyebrow = [shape[18],shape[19],shape[24],shape[25]]
    eyebrow_top_point  = ind_arr_Y_eyebrow[array_Y_eyebrow.index(min(array_Y_eyebrow))]
    image = draw_extended_line(image,eyebrow_top_point,eyebrow_top_point)
    #cv2.circle(image, shape[19], 10, (0, 0, 255), -1)
    #cv2.circle(image, shape[24], 10, (0, 0, 255), -1)
    if putTextFlag:
        cv2.putText(image, "Eyebrow top line", (eyebrow_top_point[0] - 10, eyebrow_top_point[1] - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    """
    if shape[19][0]>shape[18][0]:
        #cv2.line(image, (shape[19][0], shape[19][1]), (shape[24][0], shape[24][1]), (244, 244, 244), thickness=1)
        #image = draw_extended_line(image,shape[19],shape[24])
        #print("19: {0}, 24:{1}".format(shape[19],shape[24]))
        eyebrow_top_point = shape[19] if shape[19][1]< shape[24][1] else shape[24]
        image = draw_extended_line(image,eyebrow_top_point,eyebrow_top_point)
        #cv2.circle(image, shape[19], 10, (0, 0, 255), -1)
        #cv2.circle(image, shape[24], 10, (0, 0, 255), -1)
        if putTextFlag:
            cv2.putText(image, "Eyebrow top line", (shape[19][0] - 10, shape[19][1] - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    else:
        #cv2.line(image, (shape[18][0], shape[18][1]), (shape[25][0], shape[25][1]), (244, 244, 244), thickness=1)
        if shape[25][1]>shape[26]
        image = draw_extended_line(image,shape[18],shape[25])
        #image = draw_extended_line(image,shape[18],shape[18])
        #print("18: {0}, 25:{1}".format(shape[18],shape[25]))
        eyebrow_top_point = shape[18] if shape[18][1]< shape[25][1] else shape[25]
        image = draw_extended_line(image,eyebrow_top_point,eyebrow_top_point)
        #cv2.circle(image, shape[18], 10, (0, 0, 255), -1)
        #cv2.circle(image, shape[25], 10, (0, 0, 255), -1)
        if putTextFlag:
            cv2.putText(image, "Eyebrow top line", (shape[18][0] - 10, shape[18][1] - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    """
    # iris line
    #cv2.line(image, (center_of_left_pupil[0], center_of_left_pupil[1]), (center_of_right_pupil[0], center_of_right_pupil[1]), (244, 244, 244), thickness=1)
    p1 = center_of_left_pupil
    p2 = center_of_right_pupil
    image = draw_extended_line(image,p1,p2)
    if putTextFlag:
        cv2.putText(image, "Iris line", (p1[0] - 10, p1[1] - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    
    # lip end
    cv2.line(image, (shape[48][0], shape[48][1]), (shape[54][0], shape[54][1]), (244, 244, 244), thickness=1)
    image= draw_extended_line(image,shape[48],shape[54])
    if putTextFlag:
        cv2.putText(image, "Upper Lip line", (shape[48][0] - 10, shape[54][1] - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    #nose line
    #cv2.line(image, (nose_at_nostrils_right[0], nose_at_nostrils_right[1]), (nose_at_nostrils_left[0], nose_at_nostrils_left[1]), (244, 244, 244), thickness=1)
    #cv2.line(image, (nose_at_nostrils_right[0], nose_at_nostrils_right[1]), (nose_at_nostrils_left[0], nose_at_nostrils_left[1]), (244, 244, 244), thickness=1)
    image = draw_extended_line(image,nose_at_nostrils_right,(nose_at_nostrils_left[0],nose_at_nostrils_right[1]))
    if putTextFlag:
        cv2.putText(image, "Nose line", (nose_at_nostrils_right[0] - 10, nose_at_nostrils_right[1] - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
  
    #Face mid cutting line
    midpoint_iris = (int((center_of_left_pupil[0]+center_of_right_pupil[0])/2),int((center_of_left_pupil[1]+center_of_right_pupil[1])/2))
    #cv2.line(image, (shape[27][0], shape[27][1]), (shape[30][0], shape[30][1]), (244, 244, 244), thickness=1)
    image = draw_extended_line(image,shape[27],(shape[27][0],shape[30][1]))
    
    #face chin line
    image = draw_extended_line(image,shape[8],shape[8])
    #draw_extended_line(idx_to_coordinates[152], idx_to_coordinates[152])
    #cv2.circle(image, shape[8], 10, (0, 0, 255), -1)
    if putTextFlag:
        cv2.putText(image, "Chin line", (shape[8][0] - 10, shape[8][1] - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)    
    #hair face line
    
    #Occurences count logic - Did not work
    
    locs_mask = np.where(mask==255)
    # shape[30][0] is the x=cnst line
    #all matching points along the line
    points_locs=[]
   
    skip = False
    points_locs  = np.where(locs_mask[1]==shape[30][0])
    points_y = locs_mask[0][points_locs] 
    #find the max location in image, to find hair line 
    try:
        max_ylocation_val_in_line =points_y[-1] #np.amax(np.where(locs_mask[0]==shape[30][0]))
    except (IndexError,UnboundLocalError) as Error:
        skip=True
    try:
        if ~skip:
            hair_face_point = (
                shape[30][0],
                max_ylocation_val_in_line
                               )
            #cv2.circle(image, hair_face_point, 10, (0, 0, 255), -1)
            image = draw_extended_line(image,hair_face_point,hair_face_point)
            if putTextFlag:
                cv2.putText(image, "Hair line", (hair_face_point[0] - 10, hair_face_point[1] - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        return image,hair_face_point
    except UnboundLocalError:
        print("No Hair Line")
        return image, (shape[30][0],0)
    