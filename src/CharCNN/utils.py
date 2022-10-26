import re
import os
import nltk.data
import numpy as np
import json
import pathlib
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from string import punctuation

nltk.download("stopwords")
nltk.download("punkt")

PADDING_KEY, UNKNOWN_KEY = "<<<1234567890PADDING>>>", "<<<1234567890UNKNOWN>>>"
PROJECT_PATH = "" # pass here the path to your project
TEST_FILE_PATH = os.path.join(PROJECT_PATH, "data/", "test.txt")
TRAIN_FILE_PATH = os.path.join(PROJECT_PATH, "data/", "training.txt")
VALIDATION_FILE_PATH = os.path.join(PROJECT_PATH, "data/", "validation.txt")
KMEANS_INFO_PATH = os.path.join(PROJECT_PATH, "data/", "kmeans_info.npy")


with open(os.path.join(PROJECT_PATH, "CharCNN/", "config.json"), "r") as json_file:
    config = json.load(json_file)

os.environ["CUDA_VISIBLE_DEVICES"] = str(config["device"])

# some useful german word preprocessing algorithms
STEMMER = SnowballStemmer("german")
STOP_WORDS = set(stopwords.words("german"))
PUNCTUATION = set(punctuation)


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

# clean each document/tweet
def clean_text(document, remove_non_characters):
    # to lower-case
    res = document.lower()
    # replace punctuation with blank space
    res = "".join([chr if chr not in PUNCTUATION else ' ' for chr in res])
    if remove_non_characters:
        # replace non-characters with blank space
        res = re.sub(r"[^A-Za-zÀ-ž0-9]", ' ', res)
    else:
        # Add spaces before and after for all non-characters (for emojis)
        res = re.sub(r"[^A-Za-zÀ-ž0-9]", ' \\g<0> ', res)
    # Delete multiple spaces
    res = re.sub(r'\s+', ' ', res)

    # tokenize
    res = word_tokenize(res)
    # remove stop-words
    res = [word for word in res if word not in STOP_WORDS]
    # remove punctuation
    res = [word for word in res if word not in PUNCTUATION]
    # remove words that contain both characters and special characters (e.g: aa11bb)
    res = [word for word in res if not(word.isalnum() and not word.isalpha() and not word.isnumeric())]

    return " ".join(res)

# clean an entire corpus (a list of documents/tweets)
def clean_corpus(corpus, remove_non_characters):
    res = [clean_text(text, remove_non_characters) for text in corpus]
    return res

# split a document/tweet into words
def splitIntoWords(text):
    words = word_tokenize(text)
    words = [word for word in words if word not in STOP_WORDS]
    return words

# print mean and std of a corpus based on the frequency of the characters and words per document/tweet
def getStatistics(data, dataset_name):
    mean_len = np.mean([len(data[i][0]) for i in range(len(data))])
    std_len = np.std([len(data[i][0]) for i in range(len(data))])
    mean_words = np.mean([len(data[i][1]) for i in range(len(data))])
    std_words = np.std([len(data[i][1]) for i in range(len(data))])
    print("[{}] mean_len => {}, std => {}".format(dataset_name, mean_len, std_len))
    print("[{}] mean_word => {}, std => {}".format(dataset_name, mean_words, std_words))

# construct a character vocabulary based on the given corpus
# in order for a character to be added to this vocabulary, it's frequency must surpass a certain threshold
def constructCharVocabulary(corpus, min_threshold):
    char2idx_freq = {}
    for document in corpus:
        for char in document:
            char2idx_freq[char] = char2idx_freq.get(char, 0) + 1

    char2idx = {PADDING_KEY: 1, UNKNOWN_KEY: 2}
    for char in char2idx_freq:
        if char2idx_freq[char] >= min_threshold:
            char2idx[char] = len(char2idx) + 1

    print("Char dictionary has size: {}".format(len(char2idx)))
    return char2idx


# construct a word vocabulary based on the given corpus
# in order for a word to be added to this vocabulary, it's frequency must surpass a certain threshold
def constructWordVocabulary(corpus, min_threshold):
    word2idx_freq = {}
    for document in corpus:
        for word in document:
            # map each word to its corresponding stemming correspondation (to reduce the vocabulary size)
            if STEMMER is not None:
                word = STEMMER.stem(word)
            word2idx_freq[word] = word2idx_freq.get(word, 0) + 1

    word2idx = {PADDING_KEY: 1, UNKNOWN_KEY: 2}
    for word in word2idx_freq:
        if word2idx_freq[word] >= min_threshold:
            word2idx[word] = len(word2idx) + 1

    print("Word dictionary has size: {}".format(len(word2idx)))
    return word2idx

# converts a list of tokens to their corresponding indexes inside a dictionary
# it also does a PADDING (when the number of tokens < total_length)
# or a TRUNCATION (when the number of tokens > total_length by choosing from
# the first total_length - select_last tokens and the last select_last tokens)
def convertToIdxHelper(data, dictionary, total_length, select_last):
    if len(data) > total_length:
        return [dictionary[chr] if chr in dictionary else dictionary[UNKNOWN_KEY] for chr in
                data[: total_length - select_last] + data[-select_last:]]
    else:
        return [dictionary[chr] if chr in dictionary else dictionary[UNKNOWN_KEY] for chr in
                data] + [dictionary[PADDING_KEY]] * (total_length - len(data))

# method to write the predictions in the proper format
def writePredictions(predictions, file_path, indexes_samples):
    df = pd.DataFrame.from_dict({"id": indexes_samples, "lat": [lat for [lat, _] in predictions], "long": [long for [_, long] in predictions]})
    df.to_csv(file_path, index=False)

# searches the current directory return max(idx) + 1
# where idx is the name of each dirs in PATH. It assumes that each directory is a number
# useful for starting multiple trainings
def findNextNameDirectory(PATH):
    number = 1
    if os.path.exists(PATH):
        number = max([0] + [int(str(path).split('/')[-1]) for path in pathlib.Path(PATH).iterdir() if path.is_dir() is True and str(path).split('/')[-1].isnumeric()]) + 1
    return os.path.join(PATH, "{}".format(number))