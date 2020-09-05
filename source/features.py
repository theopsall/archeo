''' Feature Detection'''
import cv2 

def sift_detector():
    ''' Feature Extraction using SIFT'''
    sift = cv2.xfeatures2d.SIFT_create()
    pass



def surf_detector():
    ''' Feature Extraction using SURF'''
    surf = cv2.xfeatures2d.SURF_create()
    pass

def orb_detector():
    ''' Feature Extraction using ORB'''
    orb = cv2.ORB_create()
    pass