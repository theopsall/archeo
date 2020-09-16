"""
    Convolutional Neural Network on SPECTROGRAMS
"""
import os
import gc
import argparse
import numpy as np
import cv2
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.layers import (Conv2D, Dense, Flatten,
                                     MaxPooling2D, Dropout, BatchNormalization)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from source.utilities import crawl_directory, shuffling

def parse_arguments():
    """
    Command line argument parser

    Returns:
        parser: Argument Parser
    """
    record_analyze = argparse.ArgumentParser(description="Archeo Baseline")

    record_analyze.add_argument("-i", "--input", help="Input audio data")
    record_analyze.add_argument("-o", "--output",
                                default='cnn_model',
                                help="Output filename for neural network")

    return record_analyze.parse_args()

def load_unique_resized_data(tree, img_type=0):
    """
        Loading resized images and labels

        Args:
            tree (list)    : images directory
            img_type (int) : The way to read the images,
                            0           : GrayScale
                            1 (Default) : Colored
                           -1           : Unchanged

        Returns:
            images (list)  : A list which includes all the loaded images as numpy arrays
            labels (list)  : A paired list to images, containig the label for each image
    """

    labels = {}
    images = {}
    for img in tree:
        # os.sep is the system's pathname seperator
        labels[img.split(os.sep)[-1]] = []


    for img in tree:
        # os.sep is the system's pathname seperator
        labels[img.split(os.sep)[-1]].append(int(img.split(os.sep)[-2]))
        images[img.split(os.sep)[-1]] = cv2.resize(cv2.imread(img, img_type), (240, 320))

    return list(images.values()), list(labels.values())

def rescale_data(data):
    """
    Rescale image data

    Args:
        data (list): Image data

    Returns:
        rescaled [list]: Rescaled image
    """
    rescaled = np.array(data, dtype=np.float32)
    rescaled /= 255

    return rescaled

def build_model(input_size, no_classes):
    """
    Convolutional Model Architecture

    Args:
        input_size (tuple): Tuple containing images shape
        no_classes (int): Number of classes defining the number of output nodes

    Returns:
        model : The neural network model
    """

    cnn_model = Sequential()
    cnn_model.add(Conv2D(64, (3, 3), activation='relu',
                         input_shape=input_size))
    cnn_model.add(BatchNormalization())
    cnn_model.add(MaxPooling2D((3, 2)))
    # 1st Hidden Layer

    cnn_model.add(Conv2D(64, (3, 3), activation='relu'))
    cnn_model.add(BatchNormalization())
    cnn_model.add(MaxPooling2D((3, 2)))

    # 2nd Hidden Layer
    cnn_model.add(Conv2D(128, (3, 3), activation='relu'))
    cnn_model.add(MaxPooling2D((3, 2)))

    # Flatten the model
    cnn_model.add(Flatten())
    cnn_model.add(Dense(512, activation='relu'))

    # Output Layer
    cnn_model.add(Dense(no_classes, activation='sigmoid'))

    return cnn_model


if __name__ == "__main__":


    parse = parse_arguments()
    if parse.input is None:
        raise 'Input directory is Empty'
    if not os.path.isdir(parse.input):
        raise 'Input path is not directory'

    TRAIN_PATH = os.path.join(parse.input, 'TRAIN')
    TEST_PATH = os.path.join(parse.input, 'TEST')
    VAL_PATH = os.path.join(parse.input, 'VAL')

    train_tree = crawl_directory(TRAIN_PATH)
    test_tree = crawl_directory(TEST_PATH)
    val_tree = crawl_directory(VAL_PATH)


    X_train, y_train = load_unique_resized_data(train_tree)
    X_test, y_test = load_unique_resized_data(test_tree)
    X_val, y_val = load_unique_resized_data(val_tree)


    X_train, y_train = shuffling(X_train, y_train)
    X_test, y_test = shuffling(X_test, y_test)
    X_val, y_val = shuffling(X_val, y_val)

    one_hot = MultiLabelBinarizer()
    y_train = one_hot.fit_transform(y_train)
    y_test = one_hot.fit_transform(y_test)
    y_val = one_hot.fit_transform(y_val)

    classes = [str(c) for c in one_hot.classes_]
    NO_CLASSES = len(classes)

    X_train = rescale_data(X_train)
    X_test = rescale_data(X_test)
    X_val = rescale_data(X_val)

    input_shape = (240, 320, 1)

    X_train = X_train.reshape((X_train.shape[0], 240, 320, 1))
    X_val = X_val.reshape((X_val.shape[0], 240, 320, 1))
    X_test = X_test.reshape((X_test.shape[0], 240, 320, 1))

    opt = Adam(learning_rate=0.003)
    cb = EarlyStopping(patience=2, verbose=2, restore_best_weights=True)

    model = build_model(input_shape, NO_CLASSES)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])
    gc.collect()
    history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val),
                        callbacks=cb, use_multiprocessing=True, workers=4)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred.round(), target_names=classes, zero_division=1)) # V1
    model.save(parse.output+'.h5')
