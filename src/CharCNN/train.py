import tensorflow as tf
import os
import numpy as np
import json

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime
from utils import *
from model import CharCNNTrainer

with open(os.path.join(PROJECT_PATH, "CharCNN/", "config.json"), "r") as json_file:
    config = json.load(json_file)

# set the desired GPU device (useful when having multiple GPUs available)
os.environ["CUDA_VISIBLE_DEVICES"] = str(config["device"])

# parse the files to get the necessary data
indexes_train, train_data, train_labels = parseFileWithLabel(TRAIN_FILE_PATH)
indexes_validation, validation_data, validation_labels = parseFileWithLabel(VALIDATION_FILE_PATH)
indexes_test, test_data = parseFileWithoutLabel(TEST_FILE_PATH)

# extract kmeans info. Useful when config["task"] == "classification"
# as it converts the regression problem into a classification one
f = np.load(KMEANS_INFO_PATH)
train_labels_kmeans = f["cluster_label_train"]
validation_labels_kmeans = f["cluster_label_validation"]
clusters_centroids = f["clusters_centroids"]
no_classes = len(clusters_centroids)

# clean the corpus
train_data = clean_corpus(corpus=train_data, remove_non_characters=config["remove_non_characters"])
validation_data = clean_corpus(corpus=validation_data, remove_non_characters=config["remove_non_characters"])
test_data = clean_corpus(corpus=test_data, remove_non_characters=config["remove_non_characters"])

# augment the corpus by splitting each tweet into words as well.
train_data = [[sample, splitIntoWords(sample)] for sample in train_data]
validation_data = [[sample, splitIntoWords(sample)] for sample in validation_data]
test_data = [[sample, splitIntoWords(sample)] for sample in test_data]

# Print statistics for each dataset
getStatistics(train_data, dataset_name="train")
getStatistics(validation_data, dataset_name="validation")
getStatistics(test_data, dataset_name="test")

# Construct dictionaries for characters and words
CHAR2IDX_DICTIONARY = constructCharVocabulary([train_data[i][0] for i in range(len(train_data))], min_threshold=config["threshold_characters_vocabulary"])
WORD2IDX_DICTIONARY = constructWordVocabulary([train_data[i][1] for i in range(len(train_data))], min_threshold=config["threshold_words_vocabulary"])

# method to convert the data into a proper form to be fed to the model
def convert(data_list, data_labels):
    def convertChar2Idx(data):
        return convertToIdxHelper(data=data,
                                  dictionary=CHAR2IDX_DICTIONARY,
                                  total_length=config["total_characters"],
                                  select_last=config["select_last_characters"])

    def convertWord2Idx(data):
        # use stemmer for each word rather than the actual word
        return convertToIdxHelper(data=[STEMMER.stem(word) for word in data],
                                  dictionary=WORD2IDX_DICTIONARY,
                                  total_length=config["total_words"],
                                  select_last=config["select_last_words"])

    charFX = np.asarray([convertChar2Idx(data[0]) for data in data_list])
    wordFX = np.asarray([convertWord2Idx(data[1]) for data in data_list])

    data_labels = np.asarray(data_labels)
    return [charFX, wordFX], data_labels

# use the above method to convert the data
train_input, train_labels = convert(train_data, train_labels)
validation_input, validation_labels = convert(validation_data, validation_labels)
test_input, _ = convert(test_data, []) # [] is for consistency, not important


# instantiates a trainer
trainer = CharCNNTrainer(alphabet_characters_length=len(CHAR2IDX_DICTIONARY),
                         alphabet_words_length=len(WORD2IDX_DICTIONARY),
                         config=config,
                         no_classes=no_classes)

# create the path where the training info will be stored
day = datetime.now().strftime("%d_%m_%Y")
SAVED_MODELS_PATH = os.path.join(PROJECT_PATH, "CharCNN/saved_models/{}/".format(day))
os.makedirs(SAVED_MODELS_PATH, exist_ok=True)
SAVED_MODELS_PATH = findNextNameDirectory(SAVED_MODELS_PATH)
os.makedirs(SAVED_MODELS_PATH, exist_ok=True)

# save the model architecture
with open(os.path.join(SAVED_MODELS_PATH, "model_architecture.json"), "w+", encoding='utf-8') as f:
    dictionary = {}
    dictionary["model"] = vars(trainer).copy()
    del dictionary["model"]["model"]
    dictionary["config"] = config
    dictionary["char2idx_dictionary"] = CHAR2IDX_DICTIONARY
    dictionary["word2idx_dictionary"] = WORD2IDX_DICTIONARY

    json.dump(dictionary, f, indent=4)

# create callbacks to save the best model during training
task = config["task_type"]
filepath = os.path.join(SAVED_MODELS_PATH, "best_saved_model")
checkpoint = ModelCheckpoint(filepath,
                             monitor='val_mean_absolute_error' if task == "regression" else 'val_accuracy',
                             save_best_only=True,
                             save_weights_only=False,
                             verbose=1)


learning_rate_callback = LearningRateScheduler(lambda epoch, lr: max(0.0001, lr * tf.constant(config["learning_rate_decay"])))
es = EarlyStopping(monitor='val_mean_absolute_error' if task == "regression" else 'val_accuracy',
                   verbose=1,
                   patience=300,
                   min_delta=0,
                   restore_best_weights=True)

callbacks_list = [checkpoint, learning_rate_callback, es]

# fit the model
history = trainer.model.fit(train_input,
                            train_labels if task == "regression" else train_labels_kmeans,
                            validation_data=(validation_input, validation_labels if task == "regression" else validation_labels_kmeans),
                            shuffle=True,
                            batch_size=96,
                            epochs=config["epochs"],
                            verbose=1,
                            callbacks=callbacks_list)

# load the best model so far.
final_model = load_model(os.path.join(SAVED_MODELS_PATH, "best_saved_model"), compile=True)

# save training history
with open(os.path.join(SAVED_MODELS_PATH, "history.json"), "w+", encoding='utf-8') as f:
    dictionary = {}
    for key in history.history:
        dictionary[key] = str(history.history[key])
    for key in history.history:
        dictionary["minimum_{}".format(key)] = str(min(history.history[key]))
    json.dump(dictionary, f, indent=4)

# test the model on both validation and test data
if config["task_type"] == "regression":
    train_predictions = final_model.predict(train_input, batch_size=128)
    validation_predictions = final_model.predict(validation_input, batch_size=128)
    test_predictions = final_model.predict(test_input, batch_size=128)

else:
    # the model predict a cluster id. From the cluster id we select it's centroid
    train_predictions = np.argmax(final_model.predict(train_input, batch_size=128), axis=1)
    train_predictions = [clusters_centroids[cluster] for cluster in train_predictions]

    validation_predictions = np.argmax(final_model.predict(validation_input, batch_size=128), axis=1)
    validation_predictions = [clusters_centroids[cluster] for cluster in validation_predictions]

    test_predictions = np.argmax(final_model.predict(test_input, batch_size=128), axis=1)
    test_predictions = [clusters_centroids[cluster] for cluster in test_predictions]

# computes and print MAE and MSE on validation set
mae_score_validation = mean_absolute_error(validation_labels, validation_predictions)
print("mae_score on validation = {}".format(mae_score_validation))
mse_score_validation = mean_squared_error(validation_labels, validation_predictions)
print("mse_score on validation = {}".format(mse_score_validation))

# save MAE and MSE losses on validation
with open(os.path.join(SAVED_MODELS_PATH, "validation_losses.json"), "w+", encoding='utf-8') as f:
    dictionary = {"mse": mse_score_validation, "mae": mae_score_validation}
    json.dump(dictionary, f, indent=4)

# save predictions for train, validation, test on the correct format
SAVE_PREDICTIONS_PATH = os.path.join(SAVED_MODELS_PATH, "predictions")
os.makedirs(SAVE_PREDICTIONS_PATH, exist_ok=True)

writePredictions(train_predictions, os.path.join(SAVE_PREDICTIONS_PATH, "predictions_train.csv"), indexes_train)
writePredictions(validation_predictions, os.path.join(SAVE_PREDICTIONS_PATH, "predictions_validation.csv"), indexes_validation)
writePredictions(test_predictions, os.path.join(SAVE_PREDICTIONS_PATH, "predictions_test.csv"), indexes_test)