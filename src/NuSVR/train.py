import joblib
import numpy as np
from sklearn.svm import NuSVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from utils import *
import pandas as pd
import os
import json
import math

TYPE = "train"

# compute KernelNorm k_i_j_prime = k_i_j / sqrt(k_i_i * k_j_j)
def computeKernelNorm(x):
    diagonal = np.diag(x)
    n = diagonal.shape[0]
    product = np.matmul(np.reshape(diagonal, (n, 1)), np.reshape(diagonal, (1, n)))
    sqrt_matrix = np.sqrt(product)
    return x / sqrt_matrix

# read kernel matrix, normalize it and split on train, validation, test
# related paper https://www.aclweb.org/anthology/W18-3909.pdf (Andrei M. Butnaru and Radu Tudor Ionescu)
def read_kernels(train_size, validation_size):
    def extract_data(matrix_3D, row1, row2):
        return np.asarray([row[0: train_size] for row in matrix_3D[row1: row2]])

    if not os.path.exists(KERNELS_PATH_NORMALIZED):
        # kernel_matrix_full is of type [d, n, n], where n = len(train) + len(validation) + len(test)
        # and d = number of p-grams (usually chosen 6 for 3-gram, 4-gram,...,8-gram)
        print("computing kernels")
        kernels_matrix_full = np.load(KERNELS_PATH).astype(float)
        kernels_matrix_full = np.sum(kernels_matrix_full, axis=0)
        kernels_matrix_full = computeKernelNorm(kernels_matrix_full)
        np.save(KERNELS_PATH_NORMALIZED, kernels_matrix_full)
    else:
        print("loading pre-computed")
        kernels_matrix_full = np.load(KERNELS_PATH_NORMALIZED)

    # split into train, validation, test
    kernel_train = extract_data(kernels_matrix_full, 0, train_size)
    kernel_validation = extract_data(kernels_matrix_full, train_size, train_size + validation_size)
    kernel_test = extract_data(kernels_matrix_full, train_size + validation_size, len(kernels_matrix_full))

    return kernel_train, kernel_validation, kernel_test


# trainer using gridSearch method with cross validation
def train(kernel_train, train_labels, model_fname):
    estimator = NuSVR(kernel='precomputed')

    C = [0.001, 0.1, 0.5, 1.0, 5.0, 10.0]
    nu = [0.01, 0.1, 0.25, 0.5, 0.75]
    param_grid = {'C': C, 'nu': nu}

    grid_search = GridSearchCV(estimator, param_grid, cv=10, verbose=0, n_jobs=8, return_train_score=True)
    grid_search.fit(kernel_train, train_labels)

    # print best params
    print("Best parameters: ", grid_search.best_params_)
    joblib.dump(grid_search.best_estimator_, model_fname)

    # return best estimator
    return grid_search.best_estimator_

# write predictions in the correct format
def writePredictions(predictions, file_path, indexes_samples):
    df = pd.DataFrame.from_dict({"id": indexes_samples, "lat": [lat for [lat, _] in predictions],
                                 "long": [long for [_, long] in predictions]})
    df.to_csv(file_path, index=False)

if __name__ == "__main__":
    indexes_train, train_data, train_labels = parseFileWithLabel(TRAIN_FILE_PATH)
    indexes_validation, validation_data, validation_labels = parseFileWithLabel(VALIDATION_FILE_PATH)
    indexes_test, test_data = parseFileWithoutLabel(TEST_FILE_PATH)

    train_size = len(train_data)
    validation_size = len(validation_data)
    test_size = len(test_data)

    # extract kernel for train, validation, test
    kernel_train, kernel_validation, kernel_test = read_kernels(train_size=train_size, validation_size=validation_size)
    print(kernel_train.shape)
    print(kernel_validation.shape)
    print(kernel_test.shape)
    print(kernel_train)

    # for train (training the classifier) or test(loading the trained classifiers)
    if TYPE == "train":
        classifier_lat = train(kernel_train, [lat for [lat, _] in train_labels], MODEL_LAT_PATH)
        classifier_long = train(kernel_train, [long for [_, long] in train_labels], MODEL_LONG_PATH)
    else:
        classifier_lat = joblib.load(MODEL_LAT_PATH)
        classifier_long = joblib.load(MODEL_LONG_PATH)

    # Test/Predict
    y_pred_lat = classifier_lat.predict(kernel_validation)
    y_pred_long = classifier_long.predict(kernel_validation)


    mse_lat = mean_squared_error([lat for [lat, _] in validation_labels], y_pred_lat)
    mse_long = mean_squared_error([long for [_, long] in validation_labels], y_pred_long)
    mse = (mse_lat + mse_long) / 2.0

    print("mse latitude = {}".format(mse_lat))
    print("mse longitude = {}".format(mse_long))
    print("mse latitude + longitude = {}".format(mse))

    mae_lat = mean_absolute_error([lat for [lat, _] in validation_labels], y_pred_lat)
    mae_long = mean_absolute_error([long for [_, long] in validation_labels], y_pred_long)
    mae = (mae_lat + mae_long) / 2.0

    print("mae latitude = {}".format(mae_lat))
    print("mae longitude = {}".format(mae_long))
    print("mae on latitude + longitude = {}".format(mae))

    # save losses on validation set
    with open(os.path.join(PROJECT_PATH, "NuSVR/validation_losses.json"), "w+", encoding='utf-8') as f:
        dictionary = {"mse": mse, "mae": mae}
        json.dump(dictionary, f, indent=4)

    label_pred_lat = classifier_lat.predict(kernel_test)
    label_pred_long = classifier_long.predict(kernel_test)
    assert len(label_pred_lat) == len(label_pred_long)
    test_predictions = [[label_pred_lat[i], label_pred_long[i]] for i in range(len(label_pred_lat))]

    label_pred_lat = classifier_lat.predict(kernel_validation)
    label_pred_long = classifier_long.predict(kernel_validation)
    assert len(label_pred_lat) == len(label_pred_long)
    validation_predictions = [[label_pred_lat[i], label_pred_long[i]] for i in range(len(label_pred_lat))]

    # write predictions for validation and test on the correct format
    writePredictions(test_predictions, os.path.join(PROJECT_PATH, "NuSVR/test_predictions.csv"), indexes_test)
    writePredictions(validation_predictions, os.path.join(PROJECT_PATH, "NuSVR/validation_predictions.csv"), indexes_validation)

