# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 12:35:55 2021

@author: Jaideep Bommidi
"""
import cv2
def sobeldetector(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.GaussianBlur(image,(5,5),0)
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobely = cv2.Sobel(image_gray, cv2.CV_8UC1, 0, 1, ksize=5)# this will give the horizontal wrinkles in face
    sobelx = cv2.Sobel(image_gray, cv2.CV_8UC1, 1, 0, ksize=5)# this will give the vertical wrinkles in face
    return image