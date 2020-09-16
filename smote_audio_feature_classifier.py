"""
    Multi Label SVC with smoted audio features.
    Audio features are extracted with the pyAudioAnalysis.
"""
import os
import argparse
import pickle
import pandas as pd
from sklearn.svm import  LinearSVC
from sklearn.metrics import classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from pyAudioAnalysis import MidTermFeatures as mF
from source.MLSMOTE import MLSMOTE, get_minority_instace
from source.utilities import read_data, split_data

def parse_arguments():
    """
    Command line argument parser

    Returns:
        parser: Argument Parser
    """
    record_analyze = argparse.ArgumentParser(description="Archeo Baseline")

    record_analyze.add_argument("-a", "--audio", help="Input audio data")
    record_analyze.add_argument("-g", "--groundtruth", help="Ground truth data")
    record_analyze.add_argument("-o", "--output",
                                default='smoted_svc',
                                help="Output filename for classifier")

    record_analyze.add_argument("-res", "--resampled",
                                default=2000, type=int,
                                help="Number of resampled data")

    return record_analyze.parse_args()

if __name__ == "__main__":

    parse = parse_arguments()
    if parse.audio is None:
        raise 'Input directory is Empty'
    if not os.path.isdir(parse.audio):
        raise 'Input path is not a directory'
    if parse.groundtruth is None:
        raise 'Ground truth directory is Empty'
    if not os.path.isdir(parse.audio):
        raise 'Ground truth path is not a directory'

    files, labels = read_data(parse.audio, parse.groundtruth)

    one_hot = MultiLabelBinarizer()
    labels = one_hot.fit_transform(labels)
    class_names = [str(c) for c in one_hot.classes_]

    mid_window, mid_step, short_window, short_step = 1, 1, 0.1, 0.1
    f, fn, feature_names = mF.directory_feature_extraction(parse.audio,
                                                           mid_window,
                                                           mid_step,
                                                           short_window,
                                                           short_step)

    x_train, y_train, x_test, y_test = split_data(f, labels, fn)
    x_sub, y_sub = get_minority_instace(pd.DataFrame(x_train), pd.DataFrame(y_train))
    x_res, y_res = MLSMOTE(x_sub, y_sub, parse.resampled)
    print("Resampled")
    x = pd.concat([pd.DataFrame(x_train), x_res], ignore_index=True)
    y = pd.concat([pd.DataFrame(y_train), y_res], ignore_index=True)
    print('Synthetic data have been added to the train set')
    class_names = [str(c) for c in y.columns]
    print("LinearSVc Classifier")
    classifier = OneVsRestClassifier(LinearSVC(max_iter=50000, class_weight='balanced'), n_jobs=-1)
    classifier.fit(x, y)
    pickle.dump(classifier, open(parse.output + ".sav", 'wb'))

    test_yhat = classifier.predict(x_test)
    print("Testing SVM Classification Report \n{0} "
          .format(classification_report(y_test, test_yhat, target_names=class_names)))
