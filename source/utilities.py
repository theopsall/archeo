""" Utility file, containg usefull functions """
import os
import cv2
from collections import Counter
import numpy as np

def crawl_directory(directory):
    """Crawling data directory


        Args:
            directory (str) : The directory to crawl


        Returns:
            tree (list)     : A list with all the filepaths

    """
    tree = []
    subdirs = [folder[0] for folder in os.walk(directory)]

    for subdir in subdirs:
        files = next(os.walk(subdir))[2]
        for _file in files:
            tree.append(os.path.join(subdir, _file))
    return tree

def load_data(tree, img_type=0):
    """Loading images and labels


        Args:
            tree (list)    : images directory
            img_type (int) : The way to read the images,
                            0 (Default) : GrayScale
                            1           : Colored
                           -1           : Unchanged
        -
        Returns:
            images (list)  : A list which includes all the loaded images as numpy arrays
            labels (list)  : A paired list to images, containig the label for each image
    """
    labels = []
    images = []

    for img in tree:
        # os.sep is the system's pathname seperator
        labels.append(img.split(os.sep)[-2])
        # with -2 we get the label, which is always one level before the file
        images.append(cv2.imread(img, img_type))

    return images, labels

def load_unique_data(tree, img_type=0):
    """Loading unique images and labels in case of Multilabel classification

        Args:
            tree (list)    : images directory
            img_type (int) : The way to read the images,
                            0 (Default) : GrayScale
                            1           : Colored
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
        labels[img.split(os.sep)[-1]].append(img.split(os.sep)[-2])
        # with -2 we get the label, is always one level before the file
        images[img.split(os.sep)[-1]] = cv2.imread(img, img_type)

    return images.values(), labels.values()

def shuffling(images, labels) -> tuple:
    """Shuffling both images and labels


        Args:
            images (list) : List with images
            labels (list) : List with labels


        Returns:
            images (list) : A shuffled list which includes all the loaded images as numpy arrays
            labels (list) : A paired list to images, containig the label for each image
    """

    _c = list(zip(images, labels))
    np.random.shuffle(_c)
    images, labels = zip(*_c)

    return images, labels

def list_audio_files(path):
    files = []
    for file in os.listdir(path):
        if file.endswith(".wav"):
            files.append(os.path.join(path, file))
    return files


def list_gt_files(path):
    """
    [summary]

    Args:
        path ([type]): [description]

    Returns:
        [type]: [description]
    """
    files = []
    for file in os.listdir(path):
        if file.endswith(".txt"):
            files.append(os.path.join(path, file))
    return files


def is_int(s):
    """
    Check if s is integer

    Args:
        s (str): Integer in str type

    Returns:
        True: if s in Integer, False otherwise
    """
    try:
        int(s)
        return True
    except ValueError:
        return False


def read_data(path_audio, path_gt):
    """
    Read the path to wavs file and the path to labels

    Args:
        path_audio (str): Path to wavs files
        path_gt (str): Path to labels files

    Returns:
        f_a (list) : List with all the paths of audio files
        f_g (list) : List of classes for each audio file
    """
    f_a = list_audio_files(path_audio)
    f_labels = []
    f_g = list_gt_files(path_gt)
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
    """
    Splitting the data, to 80% training and 20% test. Different tracks, in order to
    avoid any bias.

    Args:
        features (list): List of the audio paths.
        labels (list): List of labels
        filenames (list): List of filenames

    Returns:
        train_audio (list): List with the audio paths for training
        train_labels (list): List with the labels for training
        test_audio (list): List with the audio paths for testing
        test_labels (list): List with the labels for testing
    """

    FILENAMES = np.random.permutation([x.split(os.sep)[-1] for x in filenames])
    TRACKS = []
    for filename in FILENAMES:
        TRACKS.append(filename.split("_")[1])

    TOTAL_FILES = len(features)
    FOR_TRAIN = int(TOTAL_FILES * 0.8)
    COUNTER = 0
    for name  in Counter(TRACKS).items():

        if COUNTER >= FOR_TRAIN:
            idx = COUNTER
        else:
            COUNTER += name[1]

    return features[:idx], labels[:idx], features[idx:], labels[idx:]

def make_dir(directory):
    """ Create a direcrory if not existing. Can get list of directories """

    if isinstance(directory, list):
        for item in directory:
            make_dir(item)
    else:
        dir_tree = directory.split("/")
        if len(dir_tree) > 0:
            full_dir_tree = [dir_tree[0]]
            for ind, _ in enumerate(dir_tree[1:]):
                full_dir_tree.append(
                    full_dir_tree[-1] + "/" + dir_tree[ind+1]
                )
            full_dir_tree = full_dir_tree[1:]
        else:
            full_dir_tree = dir_tree
        for level in full_dir_tree:
            if not os.path.exists(level):
                os.mkdir(level)

def make_dirs_with_classes(directory, no_classes):
    """ Create directory structure for given number of classes. """
    make_dir(directory)

    if isinstance(directory, list):
        for item in directory:
            make_dirs_with_classes(item, no_classes)
    else:
        for i in range(no_classes):
            make_dir(directory + str(i))

def vstack_descriptors(descriptor_list) -> list:
    """
    Stacking list of descriptors in one list

    Args:
        descriptor_list (list): List of image descriptors

    Returns:
        descriptors: Vstacked list of descriptors
    """
    descriptors = np.array(descriptor_list[0])
    for descriptor in descriptor_list[1:]:
        descriptors = np.vstack((descriptors, descriptor))

    return descriptors

def get_file_type(file_):
    """ Get file type and index for removing type from name"""
    type_ = file_[file_.rfind("."):]
    return type_, file_.rfind(".")

def crop_directory_from_file_name(file_):
    """ Returns file name without directory"""
    return file_[file_.rfind("/")+1:]

def crawl_directory_names_only(directory):
    """ Return contents of directory as a list only filenames"""
    files = crawl_directory(directory)
    subdirs = [x[0] for x in os.walk(directory)]
    tree = []
    for subdir in subdirs:
        files = next(os.walk(subdir))[2]
        if len(files) > 0:
            for _file in files:
                tree.append(_file)
    return tree


def categorize_files_by_type(file_list):
    """ Group files by type """
    types = {}
    for file_ in file_list:
        type_, _ = get_file_type(file_)
        if type_ not in types:
            types[type_] = [file_]
        else: types[type_].append(file_)
    return types

def group_directory_by_type(directory):
    """ Return directory files ordered by type """
    files_list = crawl_directory(directory)
    files_dictionary = categorize_files_by_type(files_list)
    return files_dictionary

def m4a_to_wav(file_, output_directory):
    """ Convert m4a to wav and store """
    sound = AudioSegment.from_file(file_)
    _, index = get_file_type(file_)
    clean_file_name = crop_directory_from_file_name(file_[:index])
    sound.export(output_directory+clean_file_name+".wav", format="wav")

def mp3_to_wav(file_, output_directory):
    """ Convert mp3 to wav and store """
    sound = AudioSegment.from_mp3(file_)
    _, index = get_file_type(file_)
    clean_file_name = crop_directory_from_file_name(file_[:index])
    sound.export(output_directory+clean_file_name+".wav", format="wav")
