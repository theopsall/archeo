""" Script to create label-based subdirectories for dataset with spectrograms"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from utilities import make_dirs_with_classes, crawl_directory_names_only

def parse_arguments():
    """
    Command line argument parser

    Returns:
        parser: Argument Parser
    """
    record_analyze = argparse.ArgumentParser(description="Archeo Baseline")

    record_analyze.add_argument("-a", "--audio", help="Input audio data")
    record_analyze.add_argument("-o", "--output",
                                default='SPECT',
                                help="Output filename for classifier")

    return record_analyze.parse_args()

if __name__ == '__main__':

    parse = parse_arguments()
    if parse.audio is None:
        raise 'Input directory is Empty'
    if not os.path.isdir(parse.audio):
        raise 'Input path is not a directory'

    if parse.output == 'SPECT':
        DESTINATION = os.path.join(os.getcwd(), parse.output)
    else:
        DESTINATION = parse.output

    NO_CLASSES = 13


    TRAIN_DIR = os.path.join(parse.audio, 'TRAIN') + os.sep
    VAL_DIR = os.path.join(parse.audio, 'VAL') + os.sep
    TEST_DIR = os.path.join(parse.audio, 'TEST') + os.sep
    DATA_DIRS = [TRAIN_DIR, VAL_DIR, TEST_DIR]
    make_dirs_with_classes(DATA_DIRS, NO_CLASSES)

    FILENAMES = np.random.permutation(crawl_directory_names_only(parse.audio))

    NO_CLASSES = 13

    SPECT_TRAIN_DIR = os.path.join(DESTINATION, 'TRAIN') + os.sep
    SPECT_VAL_DIR = os.path.join(DESTINATION, 'VAL') + os.sep
    SPECT_TEST_DIR = os.path.join(DESTINATION, 'TEST') + os.sep

    DATA_DIRS = [SPECT_TRAIN_DIR, SPECT_VAL_DIR, SPECT_TEST_DIR]
    make_dirs_with_classes(DATA_DIRS, NO_CLASSES)


    DESTINATION_INDEX = SPECT_TRAIN_DIR
    SOURCE_INDEX = TRAIN_DIR

    for folder in range(3):
        print(folder)
        for class_name in os.listdir(SOURCE_INDEX):
            for filename in os.listdir(os.path.join(SOURCE_INDEX, class_name)):
                dst = DESTINATION_INDEX + class_name +os.sep + filename[:-4] + ".png"
                src = SOURCE_INDEX + class_name + os.sep + filename
                # getting wav
                freq, signalData = wavfile.read(src)
                # making spectrogram
                fig = plt.figure(frameon=False)
                ax = plt.Axes(fig, [0., 0., 1., 1.])
                ax.set_axis_off()
                fig.add_axes(ax)
                if len(signalData.shape) > 1:
                    ax = plt.specgram(signalData[ :, 0], Fs=freq)
                else:
                    ax = plt.specgram(signalData, Fs=freq)
                fig.savefig(dst)
                plt.close()
        if folder == 1:
            DESTINATION_INDEX = SPECT_VAL_DIR
            SOURCE_INDEX = VAL_DIR
        else:
            DESTINATION_INDEX = SPECT_TEST_DIR
            SOURCE_INDEX = TEST_DIR
