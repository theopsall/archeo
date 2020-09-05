""" 
    Bag of Visual Words
"""
from sklearn.metrics import classification_report
from  utilities import crawl_directory, load_unique_data
from sklearn.svm import LinearSVC, SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
import Dense as grid
from Extractor import Extractor
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import os 
import gc 

def vstackDescriptors(descriptor_list):
    descriptors = np.array(descriptor_list[0])
    for descriptor in descriptor_list[1:]:
        descriptors = np.vstack((descriptors, descriptor)) 

    return descriptors

if __name__ == '__main__':
    train_path = "/home/theo/Desktop/ISMAP_2020/bovw/SPECT/TRAIN"
    test_path = "/home/theo/Desktop/ISMAP_2020/bovw/SPECT/TEST"
    val_path = "/home/theo/Desktop/ISMAP_2020/bovw/SPECT/VAL"
    train_tree = crawl_directory(val_path)
    test_tree = crawl_directory(test_path)
    X_train, y_train= load_unique_data(train_tree, img_type=0)
    X_test, y_test = load_unique_data(test_tree, img_type=0)
    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9 , 10, 11, 12] # for the classification report
    train_descriptors_list = [] 
    test_descriptors_list = []
    sift = Extractor('sift')
    
    #Training Images
    for image in X_train:
        keypoints = grid.DenseDetector(20, 20, 5).detect(image)
        kp, des = sift.compute(image, keypoints)
        train_descriptors_list.append(des)
    print("Training images descriptors has been computed ")
    
    # Testing Images
    for image in X_test:
        keypoints = grid.DenseDetector(20, 20, 5).detect(image)
        kp, des = sift.compute(image, keypoints)
        test_descriptors_list.append(des)
    print("Testing images descriptors has been computed ")
    
    train_descriptors = vstackDescriptors(train_descriptors_list)
    test_descriptors = vstackDescriptors(test_descriptors_list)
    print("Both training and testing descriptors have been vStacked")
    
    one_hot = MultiLabelBinarizer()
    y_train = one_hot.fit_transform(y_train)
    y_test = one_hot.fit_transform(y_test)
    
    k = [ 50,100,200, 500, 1000, 2000]
    
    for i, num_cluster in enumerate(k):
        print("Num of Clusters: {0}".format(num_cluster))
    
        bovw = grid.BoVW(num_clusters=num_cluster)
        print("Initiliazed Bag of Visual Words")
    
        kmeans, centroids = bovw.cluster(train_descriptors)
        print("Kmeans finished")
    
        train_X = bovw.get_feature_vector(kmeans,train_descriptors_list)
        test_X = bovw.get_feature_vector(kmeans,test_descriptors_list)
        print("Features Extracted")
    
        # Scaling Training Data 
        scale = StandardScaler().fit(train_X)        
        train_X = scale.transform(train_X)
        print("Train images normalized.") 
        # Scaling Test  Data
        scale = StandardScaler().fit(test_X)        
        test_X = scale.transform(test_X)
        print("Test images normalized.") 
        gc.collect() # Free memory to prevent killing service
        
        print("SVM Classifier for kmeans with {0} number of clusters.".format(num_cluster))
        classifier = OneVsRestClassifier(LinearSVC(max_iter=10000*(i+1)), n_jobs=-1)
        classifier.fit(train_X, y_train)
        #pickle.dump(classifier, open("SVM_K{}.sav".format(num_cluster), 'wb'))
        
        train_yhat = classifier.predict(train_X)
        test_yhat = classifier.predict(test_X)
        print(train_yhat)
        print(train_yhat[0])
        print("Training SVM Score {0} %".format(accuracy_score(y_train, train_yhat)))
        print("Testing SVM Score {0} %".format(accuracy_score(y_test, test_yhat)))
        print("Training SVM Classification Report {0} %".format(classification_report(y_train, train_yhat, labels=classes)))
        print("Testing SVM Classification Report {0} %".format(classification_report(y_test, test_yhat, labels=classes)))
        
        gc.collect() # Free memory to prevent killing service
        break
    