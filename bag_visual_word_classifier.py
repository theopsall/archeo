"""
   Experiment 1, Bag of Visual Words
"""
import os
import gc
import pickle
import argparse
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
import source.Dense as grid
from source.Extractor import Extractor
from source.utilities import crawl_directory, load_unique_data, vstack_descriptors


def parse_arguments():
    """
    Command line argument parser

    Returns:
        parser: Argument Parser
    """
    record_analyze = argparse.ArgumentParser(description="Archeo Baseline")

    record_analyze.add_argument("-i", "--input", help="Input directory with spectograms")
    record_analyze.add_argument("-o", "--output", default='bovw', help="Output filename for classifier")

    return record_analyze.parse_args()

if __name__ == '__main__':

    parse = parse_arguments()
    if parse.input is None:
        raise 'Input directory is Empty'
    if not os.path.isdir(parse.input):
        raise 'Input path is not directory'

    TRAIN_PATH = os.path.join(parse.input, 'TRAIN')
    TEST_PATH = os.path.join(parse.input, 'TEST')

    train_tree = crawl_directory(TRAIN_PATH)
    test_tree = crawl_directory(TEST_PATH)
    X_train, y_train = load_unique_data(train_tree, img_type=0)
    X_test, y_test = load_unique_data(test_tree, img_type=0)
    train_descriptors_list = []
    test_descriptors_list = []
    sift = Extractor('sift')

    #Training Images
    for image in X_train:
        keypoints = grid.DenseDetector(50, 50, 3).detect(image)
        kp, des = sift.compute(image, keypoints)
        train_descriptors_list.append(des)
    print("Training images descriptors has been computed ")

    # Testing Images
    for image in X_test:
        keypoints = grid.DenseDetector(50, 50, 3).detect(image)
        kp, des = sift.compute(image, keypoints)
        test_descriptors_list.append(des)
    print("Testing images descriptors has been computed ")

    train_descriptors = vstack_descriptors(train_descriptors_list)
    test_descriptors = vstack_descriptors(test_descriptors_list)
    print("Both training and testing descriptors have been vStacked")

    one_hot = MultiLabelBinarizer()
    y_train = one_hot.fit_transform(y_train)
    y_test = one_hot.fit_transform(y_test)
    classes = [str(c) for c in one_hot.classes_]
    k = [50, 100, 200, 500, 1000, 2000]

    for i, num_cluster in enumerate(k):
        print("Num of Clusters: {0}".format(num_cluster))

        bovw = grid.BoVW(num_clusters=num_cluster)
        print("Initiliazed Bag of Visual Words")

        kmeans, centroids = bovw.cluster(train_descriptors)
        print("Kmeans finished")

        train_X = bovw.get_feature_vector(kmeans, train_descriptors_list)
        test_X = bovw.get_feature_vector(kmeans, test_descriptors_list)
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
        pickle.dump(classifier, open(parse.output + "_{}_clusters.sav".format(num_cluster), 'wb'))

        test_yhat = classifier.predict(test_X)
        print("Testing SVM Classification Report {0} %"
              .format(classification_report(y_test, test_yhat, target_names=classes)))
        gc.collect() # Free memory to prevent killing service
