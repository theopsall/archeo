import cv2

class Extractor():
    
    def __init__(self, type_extractor):
        self.extractor = self.__extractor(type_extractor)
        
    
    def __extractor(self, type_extractor):
        
        extractor = None
        if type_extractor == 'sift':
            extractor = cv2.xfeatures2d.SIFT_create()
        elif type_extractor == 'surf':
            # At the installation process of opencv from source, you must set OPENCV_ENABLE_NONFREE CMake option in order to use SURF feature extractor,
            #  cause is patented 
            extractor = cv2.xfeatures2d.SURF_create()
        elif type_extractor == 'orb':
            extractor = cv2.ORB_create()
        else:
            print("Wrong extractor")
            
        return extractor    
        
        
        
    def compute(self, img, kps):
        # compute descriptors and keypoints
        if img is None:
            print("Not a valid Image")
            raise  TypeError
        if kps is None:
            print("Not a valid set of s")
            raise TypeError
                
        kp, des = self.extractor.compute(img, kps)
        return kp, des
