import xgboost as xgb
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
import os
from utils import *
import pandas as pd
import numpy as np
import json

combine_outputs_dict = {
    "saved_models/1/predictions/": True,
    "saved_models/2/predictions/": True,
    "saved_models/3/predictions/": True,
    "saved_models/4/predictions/": True,
    "saved_models/5/predictions/": True,
    "saved_models/6/predictions/": True,
    "saved_models/7/predictions/": True,
    "saved_models/8/predictions/": True
}

# train using gridSearch + XGBoost as estimator
def train(X, Y):
    parameters = {
        "n_estimators": [50, 100, 250],
        "max_depth": [3, 5, 7],
        'min_child_weight': [1, 3, 5],
        "learning_rate": [0.1, 0.3, 0.7],
        "subsample": [0.5, 0.75],
        'colsample_bytree': [0.5, 0.75],
        "reg_alpha": [0, 0.1, 1.0],
        "reg_lambda": [0, 0.1, 1.0]
    }

    estimator = xgb.XGBRegressor(
        objective='reg:squarederror',
        eval_metric=["mae"],
        nthread=1
    )

    grid_search = GridSearchCV(estimator=estimator,
                               param_grid=parameters,
                               scoring='neg_mean_absolute_error',
                               cv=5,
                               n_jobs=24,
                               verbose=True).fit(X, Y)
    ''' 
    grid_search = RandomizedSearchCV(estimator=estimator,
                                     param_distributions=parameters,
                                     scoring='neg_mean_absolute_error',
                                     cv=5,
                                     n_iter=300,
                                     n_jobs=24,
                                     verbose=True).fit(X, Y)
    '''
    # print best params
    print(grid_search.best_params_)

    # return best estimator
    return grid_search.best_estimator_

# compute prediction on the given samples and save in the correct format
def writePredictions(classifier_lat, classifier_long, X, file_path, indexes_samples):
    predicted_lat = classifier_lat.predict(X)
    predicted_long = classifier_long.predict(X)

    predictions = [[predicted_lat[i], predicted_long[i]] for i in range(len(predicted_lat))]

    df = pd.DataFrame.from_dict({"id": indexes_samples, "lat": [lat for [lat, _] in predictions],
                                 "long": [long for [_, long] in predictions]})
    df.to_csv(file_path, index=False)

# combine the outputs of various trained models. This new features will then be used according
# to an ensemble model based on XGBoost
if __name__ =="__main__":
    features_augmentation = False
    features_normalization = False
    use_validation_as_train = False

    indexes_train, train_data, train_labels = parseFileWithLabel(TRAIN_FILE_PATH)
    indexes_validation, validation_data, validation_labels = parseFileWithLabel(VALIDATION_FILE_PATH)
    indexes_test, test_data = parseFileWithoutLabel(TEST_FILE_PATH)

    df_train_list, df_validation_list, df_test_list = [], [], []

    # load the predictions of various trained models
    for dir in combine_outputs_dict:
        df_train = pd.read_csv(os.path.join(dir, "predictions_train.csv"))
        df_validation = pd.read_csv(os.path.join(dir, "predictions_validation.csv"))
        df_test = pd.read_csv(os.path.join(dir, "predictions_test.csv"))

        df_train_list.append(df_train)
        df_validation_list.append(df_validation)
        df_test_list.append(df_test)

    # combine these predictions
    train_features = np.array([df["lat"] for df in df_train_list] + [df["long"] for df in df_train_list]).T
    validation_features = np.array([df["lat"] for df in df_validation_list] + [df["long"] for df in df_validation_list]).T
    test_features = np.array([df["lat"] for df in df_test_list] + [df["long"] for df in df_test_list]).T

    # augment this features using polynomial method
    if features_augmentation:
        train_features = PolynomialFeatures(degree=2, interaction_only=True).fit_transform(train_features)
        validation_features = PolynomialFeatures(degree=2, interaction_only=True).fit_transform(validation_features)
        test_features = PolynomialFeatures(degree=2, interaction_only=True).fit_transform(test_features)

    # TO DO (skip this step for now)
    if features_normalization:
        pass

    # train and test on validation test (since the models might have overfit the data, i suspect that maybe
    # would be better to use only the validation test to train/test, as this predictions are closer to the "real" ones
    # than the predictions on the train data)
    if use_validation_as_train:
        X_train, X_validation, X_test = validation_features, validation_features, test_features
        Y_train_lat, Y_train_long = validation_labels[:, 0], validation_labels[:, 1]
        Y_validation_lat, Y_validation_long = validation_labels[:, 0], validation_labels[:, 1]
        indexes_train = indexes_validation
    # train and test on train set and validation test respectively
    else:
        X_train, X_validation, X_test = train_features, validation_features, test_features
        Y_train_lat, Y_train_long = train_labels[:, 0], train_labels[:, 1]
        Y_validation_lat, Y_validation_long = validation_labels[:, 0], validation_labels[:, 1]

    print("training samples: {}".format(len(X_train)))
    print("validation samples: {}".format(len(X_validation)))
    print("testing samples: {}".format(len(X_test)))
    print(X_train.shape, X_validation.shape, X_test.shape)
    print(Y_train_lat.shape, Y_train_long.shape)
    print(Y_validation_lat.shape, Y_validation_long.shape)

    # train 2 estimators for latitude/longitude
    classifier_lat = train(X_train, Y_train_lat)
    classifier_long = train(X_train, Y_train_long)

    # predict on validation
    predicted_validation_lat = classifier_lat.predict(X_validation)
    predicted_validation_long = classifier_long.predict(X_validation)

    y_score = [[predicted_validation_lat[i], predicted_validation_long[i]] for i in range(len(predicted_validation_lat))]
    y_true = validation_labels

    # compute mae/mse score on validation
    mae = mean_absolute_error(y_true, y_score)
    mse = mean_squared_error(y_true, y_score)

    print("mae score on validation = {}".format(mae))
    print("mse score on validation = {}".format(mse))

    # save mae/mse losses
    SAVED_PREDICTIONS_PATH = os.path.join(PROJECT_PATH, "Ensemble/predictions")
    os.makedirs(SAVED_PREDICTIONS_PATH, exist_ok=True)

    with open(os.path.join(SAVED_PREDICTIONS_PATH, "validation_losses.json"), "w+", encoding='utf-8') as f:
        dictionary = {"mse": mse, "mae": mae}
        json.dump(dictionary, f, indent=4)

    # save predictions for train, validation, test on the correct format
    writePredictions(classifier_lat, classifier_long, X_train, os.path.join(SAVED_PREDICTIONS_PATH, "predictions_train.csv"), indexes_train)
    writePredictions(classifier_lat, classifier_long, X_validation, os.path.join(SAVED_PREDICTIONS_PATH, "predictions_validation.csv"), indexes_validation)
    writePredictions(classifier_lat, classifier_long, X_test, os.path.join(SAVED_PREDICTIONS_PATH, "predictions_test.csv"), indexes_test)
