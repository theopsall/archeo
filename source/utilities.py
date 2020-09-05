""" Auxiliary file, with helper functions """

import numpy as np 
import os 
import cv2





def crawl_directory(directory) -> list:
    """Crawling data directory
        
        ------------------------------
        Parameters:
            directory (str) : The directory to crawl

        ------------------------------
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

def load_data(tree, img_type = 0) -> tuple :
    """Loading images and labels
    
        ------------------------------
        Parameters:
            tree (list)    : images directory
            img_type (int) : The way to read the images, 
                            0 (Default) : GrayScale
                            1           : Colored
                           -1           : Unchanged
        ------------------------------
        Returns:
            images (list)  : A list which includes all the loaded images as numpy arrays
            labels (list)  : A paired list to images, containig the label for each image
    """
    labels = []
    images = []
    
    for img in tree:
        # os.sep is the system's pathname seperator 
        labels.append(img.split(os.sep)[-2]) # with -2 we get the label, which is always one level before the file
        images.append(cv2.imread(img, img_type))
        

    return images, labels

def load_unique_data(tree, img_type = 0) -> tuple :
    """Loading unique images and labels in case of Multilabel classification
    
        ------------------------------
        Parameters:
            tree (list)    : images directory
            img_type (int) : The way to read the images, 
                            0 (Default) : GrayScale
                            1           : Colored
                           -1           : Unchanged
        ------------------------------
        Returns:
            images (list)  : A list which includes all the loaded images as numpy arrays
            labels (list)  : A paired list to images, containig the label for each image
    """
    print(len(tree))
    labels = {}
    images = {}
    for img in tree:
        # os.sep is the system's pathname seperator 
        labels[img.split(os.sep)[-1]] = []

    
    for img in tree:
        # os.sep is the system's pathname seperator 
        labels[img.split(os.sep)[-1]].append(img.split(os.sep)[-2]) #  with -2 we get the label, which is always one level before the file        
        images[img.split(os.sep)[-1]] = cv2.imread(img, img_type)
        

    return images.values(), labels.values()

def shuffling(images, labels) -> tuple :
    """Shuffling both images and label
    
        ------------------------------
        Parameters:
            images (list) : List with images
            labels (list) : List with labels, 
        
        ------------------------------
        Returns:
            images (list) : A shuffled list which includes all the loaded images as numpy arrays
            labels (list) : A paired list to images, containig the label for each image
    """
    
    c = list(zip(images, labels))
    np.random.shuffle(c)
    images, labels = zip(*c)
    
    
    return images, labels
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

def crawl_directory(directory):
    """ Return contents of directory as a list """
    subdirs = [x[0] for x in os.walk(directory)]
    tree = []
    for subdir in subdirs:
        files = next(os.walk(subdir))[2]
        if len(files) > 0:
            for _file in files:
                tree.append(subdir + "/" + _file)
    return tree

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
    