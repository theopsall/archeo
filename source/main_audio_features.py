""" 
    Bag of Visual Words with audio features
"""
from pyAudioAnalysis import ShortTermFeatures as sF
from pyAudioAnalysis import MidTermFeatures as mF
from pyAudioAnalysis import audioTrainTest as aT
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.svm import LinearSVC, SVC
from  utilities import crawl_directory, load_unique_image_data, vstackDescriptors, get_audio_paths, crawl_directory_names_only
from Extractor import Extractor
from collections import Counter
import scipy.io.wavfile as wavfile
import Dense as grid
import numpy as np
import argparse
import pickle
import cv2
import os 
import gc 

def parse_arguments():
    record_analyze = argparse.ArgumentParser(description="Archeo Baseline")

    record_analyze.add_argument("-a", "--audio", help="Input audio data")
    record_analyze.add_argument("-g", "--groundtruth", help="Ground truth data")
    record_analyze.add_argument("--balance", action="store_true",
                                  help="Make Balanced")
    return record_analyze.parse_args()

def to_list_of_features(feat_mat, feat_fn, fn, labels):
    list_of_features_per_class = []
    classes = []
    for iF, f in enumerate(feat_mat):
        print(f.shape, feat_fn[iF])
        # match feature extraction filenames to gt filenames
        if feat_fn[iF] in fn:
            print("theo ; >",feat_fn[iF])
            print("FN",fn.index(feat_fn[iF]))
            cur_classes = labels[fn.index(feat_fn[iF])]
            for c in cur_classes:
                if c in classes:
                    list_of_features_per_class[classes.index(c)].append(f)
                else:
                    list_of_features_per_class.append([f])
                    classes.append(c)
    print(classes)
    print(len(list_of_features_per_class))

    for iF in range(len(list_of_features_per_class)):
        list_of_features_per_class[iF] = np.array(list_of_features_per_class[iF])

    for f in list_of_features_per_class:
        print(f.shape)
    return list_of_features_per_class, classes

def list_audio_files(path):
    files = []
    for file in os.listdir(path):
        if file.endswith(".wav"):
            files.append(os.path.join(path, file))
    return files


def list_gt_files(path):
    files = []
    for file in os.listdir(path):
        if file.endswith(".txt"):
            files.append(os.path.join(path, file))
    return files


def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def read_data(path_audio, path_gt):
    f_a = list_audio_files(path_audio)
    f_labels = []
    f_g = list_gt_files(path_gt)
    print(len(f_a))
    print(len(f_g))

    for f in f_a:
        cur_gt = os.path.join(path_gt, os.path.basename(f.replace(".wav",
                                                                  ".txt")))
        if cur_gt in f_g:
            with open(cur_gt) as f_gt:
                str = f_gt.read()
                nums = (str.replace("\n", "").split(" "))
                #print(nums)
                labels = [int(l) for l in nums if is_int(l)]
                #print(labels)
                f_labels.append(labels)

    c = list(zip(f_a, f_labels))
    c = sorted(c)
    f_a, f_labels = zip(*c)
    return f_a, f_labels

def split_data(features, labels, filenames):
    
    FILENAMES = np.random.permutation([x.split(os.sep)[-1] for x in filenames])
    TRACKS = []
    for filename in FILENAMES:
        TRACKS.append(filename.split("_")[1])

    TOTAL_FILES = len(features)
    FOR_TRAIN = int(TOTAL_FILES * 0.8)
    COUNTER = 0
    for name  in Counter(TRACKS).items():

        if COUNTER => FOR_TRAIN :
            print("COUNTER", COUNTER)
            idx = COUNTER
        else :
            COUNTER += name[1]


    return features[:idx], labels[:idx], features[idx:], labels[idx:]


if __name__ == '__main__':
    wavs = "/home/theo/Desktop/ISMAP_2020/bovw/raw_data_and_labels/DATA_SOURCE/"
    labs = "/home/theo/Desktop/ISMAP_2020/bovw/raw_data_and_labels/DATA_LABELS/"

    files, labels = read_data(wavs, labs)

    one_hot = MultiLabelBinarizer()
    labels = one_hot.fit_transform(labels)
    class_names = [str(c) for c in one_hot.classes_]

    mid_window, mid_step, short_window, short_step = 1, 1, 0.1, 0.1
    f, fn, feature_names = mF.directory_feature_extraction(wavs,
                                                           mid_window,
                                                           mid_step,
                                                           short_window,
                                                           short_step)

    X_train, y_train, X_test, y_test = split_data(f, labels, fn)
    print(len(X_train))
    print(len(X_test))
    #list_of_features, class_names = to_list_of_features(f, fn, files, labels)
    #class_names = [str(c) for c in class_names]
    
    print("LinearSVc Classifier")
    classifier = OneVsRestClassifier(LinearSVC(max_iter=10000), n_jobs=-1)
    classifier.fit(X_train, y_train)
    #pickle.dump(classifier, open("SVM_K{}.sav".format(num_cluster), 'wb'))
    
    train_yhat = classifier.predict(X_train)
    test_yhat = classifier.predict(X_test)
    print(train_yhat)
    print(train_yhat[0])
    print("Training SVM Score {0} %".format(accuracy_score(y_train, train_yhat)))
    print("Testing SVM Score {0} %".format(accuracy_score(y_test, test_yhat)))
    print("Training SVM Classification Report {0} %".format(classification_report(y_train, train_yhat, labels=class_names)))
    print("Testing SVM Classification Report {0} %".format(classification_report(y_test, test_yhat, labels=class_names)))
        
   