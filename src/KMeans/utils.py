import os
import numpy as np

PADDING_KEY, UNKNOWN_KEY = "<<<1234567890PADDING>>>", "<<<1234567890UNKNOWN>>>"
PROJECT_PATH = "" # pass here the path to your project

TEST_FILE_PATH = os.path.join(PROJECT_PATH, "data/", "test.txt")
TRAIN_FILE_PATH = os.path.join(PROJECT_PATH, "data/", "training.txt")
VALIDATION_FILE_PATH = os.path.join(PROJECT_PATH, "data/", "validation.txt")
KMEANS_INFO_PATH = os.path.join(PROJECT_PATH, "data/", "kmeans_info.npy")


def parseFileWithLabel(FILE_PATH):
    with open(FILE_PATH, "r") as f:
        data = f.read().splitlines()
        features = [splitted_line[3] for splitted_line in
                    [line.split(",", maxsplit=3) for line in data]]
        labels = np.array([[float(splitted_line[1]), float(splitted_line[2])] for splitted_line in
                  [line.split(",", maxsplit=3) for line in data]])

    return features, labels


def parseFileWithoutLabel(FILE_PATH):
    with open(FILE_PATH, "r") as f:
        data = f.read().splitlines()
        indexes = np.array([int(splitted_line[0]) for splitted_line in
                    [line.split(",", maxsplit=1) for line in data]])
        features = [splitted_line[1] for splitted_line in
                    [line.split(",", maxsplit=1) for line in data]]
    return indexes, features

