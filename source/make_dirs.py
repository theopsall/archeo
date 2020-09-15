""" Script to create label-based subdirectories for dataset """

import os
import shutil
import argparse
from collections import Counter
import numpy as np
from utilities import make_dirs_with_classes, crawl_directory_names_only


if __name__ == '__main__':
    DATASET = "/home/theo/Desktop/ISMAP_2020/bovw/raw_data_and_labels/DATA_SOURCE/"
    LABELS = "/home/theo/Desktop/ISMAP_2020/bovw/raw_data_and_labels/DATA_LABELS/"
    NO_CLASSES = 13

    TRAIN_DIR = "/home/theo/Desktop/ISMAP_2020/bovw/WAVES/TRAIN/"
    VAL_DIR = "/home/theo/Desktop/ISMAP_2020/bovw/WAVES/VAL/"
    TEST_DIR = "/home/theo/Desktop/ISMAP_2020/bovw/WAVES/TEST/"
    DATA_DIRS = [TRAIN_DIR, VAL_DIR, TEST_DIR]
    make_dirs_with_classes(DATA_DIRS, NO_CLASSES)

    FILENAMES = np.random.permutation(crawl_directory_names_only(DATASET))


    TRACKS = []
    for filename in FILENAMES:
        TRACKS.append(filename.split("_")[1])

    TOTAL_FILES = len(FILENAMES)
    FOR_TRAIN = int(TOTAL_FILES * 0.8)
    FOR_VAL = int(TOTAL_FILES * 0.1)
    FOR_TEST = int(TOTAL_FILES - FOR_TRAIN - FOR_VAL)

    DIRECTORY_INDEX = TRAIN_DIR
    COUNTER = 0

    for name  in Counter(TRACKS).items():
        COUNTER += name[1]

        if COUNTER > FOR_TRAIN + FOR_VAL:
            DIRECTORY_INDEX = TEST_DIR

        elif COUNTER > FOR_TRAIN:
            DIRECTORY_INDEX = VAL_DIR

        for filename in FILENAMES:
            if name[0] == filename.split('_')[1]:
                with open(LABELS + filename.split(".")[0] + ".txt", "r") as l:
                    line = l.readline().rstrip()
                    for label in line.split(' '):
                        if label.isnumeric():
                            new_directory = DIRECTORY_INDEX + label + os.path.sep + filename
                            shutil.copyfile(DATASET + os.path.sep + filename, new_directory)
