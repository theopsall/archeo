import sys
import cv2
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from source.Extractor import Extractor
# visual dictionary is also known as codebook

class DenseDetector():
    """
        Dense Detector parse image in grid.
    """

    def __init__(self, step_size=20, feature_scale=20, img_bound=20) :

        self.initXyStep = step_size # size of the circle
        self.initFeatureScale = feature_scale # distance between the circles
        self.initImgBound = img_bound # Starting point from top left corner


    def detect(self, img) :
        """Detects the keypoints of the image in grid

        Args:
            img (list): Image in Array

        Returns:
            keypoints [list]: Returns the Grid Keypoints of the image
        """
        # Detect keypoints
        keypoints = []
        rows, cols = img.shape[:2]
        #self.initXyStep = int(rows/self.initFeatureScale)
        for x in range(self.initImgBound, rows, self.initFeatureScale):
            for y in range(self.initImgBound, cols, self.initFeatureScale):
                keypoints.append(cv2.KeyPoint(float(y), float(x), self.initXyStep))
        return keypoints



class BoVW(object):
    """
        Bag of Visual Words
    """

    def __init__(self, num_clusters=32):
        self.num_dims = 18
        self.extractor = Extractor('sift')
        self.num_clusters = num_clusters
        self.num_retries = 10

    def cluster(self, datapoints):

        kmeans = MiniBatchKMeans(self.num_clusters, init_size= 3*self.num_clusters+100)
        res = kmeans.fit(datapoints)
        centroids = res.cluster_centers_

        return kmeans, centroids

    def normalize(self, input_data):

        sum_input = np.sum(input_data)
        if sum_input > 0:
            return input_data / sum_input
        else :
            return input_data

    def get_feature_vector(self, kmeans, descriptors):

        im_features = np.array([np.zeros(self.num_clusters) 
                                for i in range(len(descriptors))])

        for i in range(len(descriptors)):

            for j in range(len(descriptors[i])):
                feature = descriptors[i][j]
                feature = feature.reshape(1,128)
                idx = kmeans.predict(feature)
                im_features[i][idx] += 1

        return im_features
