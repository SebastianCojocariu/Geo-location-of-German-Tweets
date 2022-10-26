import os
import numpy as np

PROJECT_PATH = "" # pass here the path to your project
TEST_FILE_PATH = os.path.join(PROJECT_PATH, "data/", "test.txt")
TRAIN_FILE_PATH = os.path.join(PROJECT_PATH, "data/", "training.txt")
VALIDATION_FILE_PATH = os.path.join(PROJECT_PATH, "data/", "validation.txt")
KERNELS_PATH = os.path.join(PROJECT_PATH, "data/sentences/kernels/all_sentences_without_cleaning_presence_3_8.npy")
KERNELS_PATH_NORMALIZED = os.path.join(PROJECT_PATH, "data/sentences/kernels/all_sentences_without_cleaning_presence_3_8_addition_normalized.npy")
MODEL_LAT_PATH = os.path.join(PROJECT_PATH, "NuSVR/model_lat")
MODEL_LONG_PATH = os.path.join(PROJECT_PATH, "NuSVR/model_long")

# parse training and validation files
def parseFileWithLabel(FILE_PATH):
    with open(FILE_PATH, "r") as f:
        data = f.read().splitlines()
        indexes = np.array([float(splitted_line[0]) for splitted_line in
                  [line.split(",", maxsplit=3) for line in data]])
        features = [splitted_line[3] for splitted_line in
                    [line.split(",", maxsplit=3) for line in data]]
        labels = np.array([[float(splitted_line[1]), float(splitted_line[2])] for splitted_line in
                  [line.split(",", maxsplit=3) for line in data]])

    return indexes, features, labels

# parse test file
def parseFileWithoutLabel(FILE_PATH):
    with open(FILE_PATH, "r") as f:
        data = f.read().splitlines()
        indexes = np.array([int(splitted_line[0]) for splitted_line in
                    [line.split(",", maxsplit=1) for line in data]])
        features = [splitted_line[1] for splitted_line in
                    [line.split(",", maxsplit=1) for line in data]]
    return indexes, features

