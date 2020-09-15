""" Script to create label-based subdirectories for dataset """

import os
import shutil
import argparse
from collections import Counter
import numpy as np
from utilities import make_dirs_with_classes, crawl_directory_names_only

def parse_arguments():
    """
    Command line argument parser

    Returns:
        parser: Argument Parser
    """
    record_analyze = argparse.ArgumentParser(description="Archeo Baseline")

    record_analyze.add_argument("-a", "--audio", help="Input audio data")
    record_analyze.add_argument("-g", "--groundtruth", help="Groundtruth of audio data")
    record_analyze.add_argument("-o", "--output",
                                default='WAVS',
                                help="Output filename for classifier")

    return record_analyze.parse_args()

if __name__ == '__main__':

    parse = parse_arguments()
    if parse.audio is None:
        raise 'Input directory is Empty'
    if not os.path.isdir(parse.audio):
        raise 'Input path is not a directory'
    if parse.groundtruth is None:
        raise 'Ground truth directory is Empty'
    if not os.path.isdir(parse.audio):
        raise 'Ground truth path is not a directory'

    NO_CLASSES = 13

    if parse.output == 'WAVS':
        DESTINATION = os.path.join(os.getcwd(), parse.output)
    else:
        DESTINATION = parse.output

    TRAIN_DIR = os.path.join(DESTINATION, 'TRAIN') + os.sep
    VAL_DIR = os.path.join(DESTINATION, 'VAL') + os.sep
    TEST_DIR = os.path.join(DESTINATION, 'TEST') + os.sep
    DATA_DIRS = [TRAIN_DIR, VAL_DIR, TEST_DIR]
    make_dirs_with_classes(DATA_DIRS, NO_CLASSES)

    FILENAMES = np.random.permutation(crawl_directory_names_only(parse.audio))


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
                with open(parse.groundtruth + filename.split(".")[0] + ".txt", "r") as l:
                    line = l.readline().rstrip()
                    for label in line.split(' '):
                        if label.isnumeric():
                            new_directory = DIRECTORY_INDEX + label + os.sep+ filename
                            shutil.copyfile(parse.audio +os.sep+ filename, new_directory)
