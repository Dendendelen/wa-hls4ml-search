import os

import sklearn.model_selection

if os.system('nvidia-smi') == 0:
    import setGPU

import glob
import sys
import argparse

import math
from bisect import bisect_left
import datetime

import numpy as np
import pandas as pd

import yaml
import csv
import json
import pickle

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras.models import load_model

import keras
import keras.backend as K
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

import wa_hls4ml_model

from callbacks import all_callbacks

if os.system('nvidia-smi') == 0:
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    device = "/GPU:0"
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
else:
    device = "/CPU:0"


def stack_dict(inputs, fun=tf.stack):
    values = []
    for key in sorted(inputs.keys()):
        values.append(tf.cast(inputs[key], tf.float32))

    return fun(values, axis=-1)

# d_in,	d_out, prec, rf, strategy
#hls_synth_success, TargetClockPeriod_hls,	WorstLatency_hls, IntervalMax_hls, FF_hls, LUT_hls, BRAM_18K_hls, DSP_hls

def preprocess(csv_file, input_features, output_features, _binary_feature_names, _numeric_feature_names, _categorical_feature_names):
    # Step 1: Read the CSV file
    df = pd.read_csv(csv_file)
    df.fillna(-1)
    preprocessed_data = []
    processing_input = True
    for sel_feature_names in [input_features, output_features, input_features]: # Do input features twice to get normalized and non-normalized values
        binary_feature_names = [item for item in _binary_feature_names if item in sel_feature_names]
        numeric_feature_names = [item for item in _numeric_feature_names if item in sel_feature_names]
        categorical_feature_names = [item for item in _categorical_feature_names if item in sel_feature_names]

        # Step 2: Split the DataFrame into input and output DataFrames
        input_data = df[sel_feature_names]

        # Steps 3-6: Process binary, numeric, and categorical features
        preprocessed_inputs = preprocess_features(input_data, binary_feature_names, numeric_feature_names, categorical_feature_names, processing_input)
        processing_input = False

        # Step 7: Convert the preprocessed data to numpy arrays
        preprocessed_inputs = preprocessed_inputs.numpy()
        preprocessed_data.append(preprocessed_inputs)

    return np.nan_to_num(preprocessed_data[0], nan=-1), np.nan_to_num(preprocessed_data[1], nan=-1), np.nan_to_num(preprocessed_data[2], nan=-1)

def preprocess_features(data, binary_feature_names, numeric_feature_names, categorical_feature_names,processing_input=True):
    preprocessed = []

    # Step 4: Process numeric features
    if numeric_feature_names:
        for name in numeric_feature_names:
            data[name] = pd.to_numeric(data[name], errors='coerce')
        print(data[numeric_feature_names].head())
        if processing_input:
            normalizer = tf.keras.layers.Normalization(axis=-1)
            normalizer.adapt(data[numeric_feature_names])
            numeric_normalized = normalizer(data[numeric_feature_names])
        else:
            numeric_normalized = data[numeric_feature_names]
        preprocessed.append(numeric_normalized)

    # Step 3: Process binary features
    if binary_feature_names:
        for name in binary_feature_names:
            value = tf.cast(data[name].astype(bool), tf.float32)
            value = tf.reshape(value, [-1, 1])
            preprocessed.append(value)

    # Step 5: Process categorical features
    if categorical_feature_names:
        for name in categorical_feature_names:
            vocab = sorted(set(data[name][1:])) #Exclude header
            if type(vocab[0]) is str:
                lookup = tf.keras.layers.StringLookup(vocabulary=vocab, output_mode='one_hot', num_oov_indices=0)
            else:
                lookup = tf.keras.layers.IntegerLookup(vocabulary=vocab, output_mode='one_hot', num_oov_indices=0)
            one_hot = lookup(data[name])
            print(data[name].shape)
            print(one_hot)
            preprocessed.append(one_hot)

    # Step 6: Concatenate all processed features
    preprocessed_data = tf.concat(preprocessed, axis=-1)

    return preprocessed_data

def plot_loss(name, history):
    plt.figure(figsize=(10,6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.savefig(name+'_train_loss.png')
    plt.show()

def plot_histograms(y_predicted, y_actual, X_raw):
    plt.figure(figsize=(10,6))

    y_difference = y_actual - y_predicted
    y_abs_diff = np.abs(y_difference)



    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.savefig(name+'_train_loss.png')
    plt.show()


def plot_results(name, mpl_plots, y_test, y_pred, X_raw_test, output_features):

    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'orange']


    if mpl_plots:
        # Iterate over each column
        print(y_test[0])
        print(y_test[1])
        for i in range(y_test.shape[1]):
            plt.figure(figsize=(10, 6))  # Create a new figure for each plot
            print(output_features[i])
            plt.scatter(y_test[:, i], y_pred[:, i], s=20, label=output_features[i])
            plt.title('Actual vs Predicted for ' + output_features[i])
            plt.xlabel('Actual Value')
            plt.ylabel('Predicted Value')
            plt.legend()

            # Plot a diagonal (for reference)
            plt.plot([np.min(y_test[:, i]), np.max(y_test[:, i])], [np.min(y_test[:, i]), np.max(y_test[:, i])], 'r')
            plt.tight_layout()
            plt.savefig(output_features[i] + '_predicted_vs_true.png')
            plt.show()


    import plotly.graph_objects as go
    import plotly.io as pio
    import chart_studio.plotly as py

    # Create a figure
    fig = go.Figure()

    marker_shapes = {0: 'star', 1: 'square'}
    strat_dict = {0: 'Latency', 1: 'Resource'}

    # Calculate the overall min and max values for the reference line
    overall_min = np.min(y_test)
    overall_max = np.max(y_test)
    n_features = len(output_features)
    n_cols = 2
    n_rows = math.ceil(n_features / n_cols)

    # Create a subplot
    import plotly.subplots as sp
    fig = sp.make_subplots(rows=n_rows, cols=n_cols,
                           vertical_spacing=0.03, horizontal_spacing=0.03,
                            x_title='Actual Value',
                            y_title='Predicted Value',
                           subplot_titles=output_features)

    # Iterate over each column
    for i in range(n_features):
        # Calculate the current row and column
        row = i // n_cols + 1
        col = i % n_cols + 1

        # Iterate over each strategy
        for strategy in [0, 1]:
            # Create a mask for the current strategy
            mask = X_raw_test[:,
                   strategy + 4] == 1  # assuming the 'strategy' is the 5th and 6th feature in your input data
            text_arr = [
                f"{int(point[2])}-bit {int(point[0])}x{int(point[1])} @ RF={int(point[3])} ({strat_dict[strategy]})" for
                point in X_raw_test if point[strategy + 4] == 1]

            # Create a scatter plot for each output feature and strategy
            scatter = go.Scatter(
                x=y_test[mask, i],
                y=y_pred[mask, i],
                mode='markers',
                name=f'{output_features[i]} - {strat_dict[strategy]}',
                legendgroup=f'{output_features[i]}',
                marker=dict(
                    symbol=marker_shapes[strategy],  # Use different marker shapes for different strategies
                    color=colors[i],
                    size=10,
                    opacity=0.7,
                ),
                hovertemplate=
                '%{text}<br>' +
                '<i>Actual</i>: %{x}<br>' +
                '<b>Predicted</b>: %{y}<br><extra></extra>',
                text=text_arr
            )
            # Add the scatter plot to the subplot
            fig.add_trace(scatter, row=row, col=col)
        #fig.update_yaxes(scaleanchor=f"x{col}", scaleratio=1, row=row, col=col)

        # Add a reference line
        fig.add_trace(
            go.Scatter(
                x=[np.min(y_test[:, i]), np.max(y_test[:, i])],
                y=[np.min(y_test[:, i]), np.max(y_test[:, i])],
                mode='lines',
                line=dict(color='black'),
                showlegend=False
            ),
            row=row, col=col
        )

    # Set the layout
    fig.update_layout(height=1900, width=1900, title='wa-hls4ml 1-Layer Dense Toy Model - Actual vs Predicted')

    pio.write_html(fig, file='plots/'+name+'_wa-hls4ml_outputs.html', auto_open=False)
    # py.plot(fig, filename='wa-hls4ml_outputs', auto_open=False)
    # Show the plot
    # fig.show()


def preprocess_data():
    input_features = ["d_in", "d_out", "prec", "rf", "strategy", "hls_synth_success"]
    output_features = ["WorstLatency_hls", "IntervalMax_hls", "FF_hls", "LUT_hls", "BRAM_18K_hls",
                       "DSP_hls"]
    binary_feature_names = ['hls_synth_success']
    numeric_feature_names = ["d_in", "d_out", "prec", "rf", "WorstLatency_hls", "IntervalMax_hls", "FF_hls", "LUT_hls",
                             "BRAM_18K_hls", "DSP_hls"]
    categorical_feature_names = ["strategy"]

    X, y, X_raw = preprocess('results/results_format_test.csv', input_features, output_features,
                             binary_feature_names, numeric_feature_names,
                             categorical_feature_names)


    # Split the data 70 - 20 - 10 train test val
    # Train and test
    print(X.shape, y.shape)
    print("X Data: ",input_features)
    print(X)
    print("Y Data: ",output_features)
    print(y[0])
    return sklearn.model_selection.train_test_split(X, y, X_raw,  test_size=0.2, random_state=42, shuffle=True)

def train_classifier(X_train, y_train):

    model = wa_hls4ml_model.create_model_classification()

    adam = AdamW(learning_rate=0.001)
    model.compile(optimizer=adam, loss=tf.keras.losses.Hinge(), metrics=['mse', 'mae'])
    callbacks = all_callbacks(
        stop_patience=1000,
        lr_factor=0.5,
        lr_patience=7,
        lr_epsilon=0.000001,
        lr_cooldown=2,
        lr_minimum=0.00000001,
        outputDir='model_class_1',
    )
    history = model.fit(
        X_train,
        y_train,
        batch_size=256,
        epochs=150,
        validation_split=0.125,
        shuffle=True,
        callbacks=callbacks.callbacks,
    )
    plot_loss('classifier', history)


def train_regressor(X_succeeded_train, y_succeeded_train):

    model = wa_hls4ml_model.create_model_regression()

    adam = AdamW(learning_rate=0.005)
    model.compile(optimizer=adam, loss=tf.keras.losses.Huber(), metrics=['mse', 'mae'])
    callbacks = all_callbacks(
        stop_patience=1000,
        lr_factor=0.5,
        lr_patience=8,
        lr_epsilon=0.0001,
        lr_cooldown=3,
        lr_minimum=0.0000000001,
        outputDir='model_reg_1',
    )
    history = model.fit(
        X_succeeded_train,
        y_succeeded_train,
        batch_size=128,
        epochs=350,
        validation_split=0.125,
        shuffle=True,
        callbacks=callbacks.callbacks,
    )
    plot_loss('regression',history)

def display_results_classifier(X_test, X_raw_test, y_test, output_features):
    
    model = load_model('model_class_1/KERAS_check_best_model.h5')

    # Predict the output for X_test
    y_pred = model.predict(X_test)

    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    y_pred_binary = np.where(y_pred > 1, 1, 0)
    confusion = confusion_matrix(y_pred_binary, y_test)

    print(f'Mean Absolute Error (MAE): {mae}')
    print(f'Mean Squared Error (MSE): {mse}')
    print(f'Root Mean Squared Error (RMSE): {rmse}')
    print(f'R-squared (R2 Score): {r2}')
    print('Confusion matrix:')
    print(confusion)

    y_test_2d = np.reshape(y_test, (y_test.shape[0], 1))

    # plot our classification results
    plot_results("classifier", False, y_test_2d, y_pred, X_raw_test, output_features)

def display_results_regressor(X_succeeded_test, y_succeeded_test, X_raw_succeeded_test, output_features):

    model = load_model('model_reg_1/KERAS_check_best_model.h5')

    # Predict the output for X_test
    y_pred = model.predict(X_succeeded_test)

    # Calculate metrics
    mae = mean_absolute_error(y_succeeded_test, y_pred)
    mse = mean_squared_error(y_succeeded_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_succeeded_test, y_pred)

    print(f'Mean Absolute Error (MAE): {mae}')
    print(f'Mean Squared Error (MSE): {mse}')
    print(f'Root Mean Squared Error (RMSE): {rmse}')
    print(f'R-squared (R2 Score): {r2}')

    # plot our regression results
    plot_results("regression", False, y_succeeded_test, y_pred, X_raw_succeeded_test, output_features)

if __name__ == "__main__":

    train = False

    # get raw data out
    X_train, X_test, y_train, y_test, X_raw_train, X_raw_test = preprocess_data()

    # separate the classification task data - remove the classification from input data
    X_train_classifier = np.append(X_train[:, :4], X_train[:, 5:], axis=1)
    X_test_classifier = np.append(X_test[:, :4], X_test[:, 5:], axis=1)
    X_raw_test_classifier = np.append(X_raw_test[:, :4], X_raw_test[:, 5:], axis=1)

    # get just the classification task as its own variable
    y_train_classifier = X_train[:, 4]
    y_test_classifier = X_test[:, 4]

    # train the classifier
    print("Training the classifier...")
    # train_classifier(X_train_classifier, y_train_classifier)
    display_results_classifier(X_test_classifier, X_raw_test_classifier, y_test_classifier, ["hls_synth_success"])

    # find which synth has succeeded
    succeeded_synth_gt_test = np.nonzero(y_test_classifier)
    succeeded_synth_gt_train = np.nonzero(y_train_classifier)

    # only test on successes
    X_succeeded_test = X_test[succeeded_synth_gt_test]
    y_succeeded_test = y_test[succeeded_synth_gt_test]
    X_raw_succeeded_test = X_raw_test[succeeded_synth_gt_test]

    # only train on successes
    X_succeeded_train = X_train[succeeded_synth_gt_train]
    y_succeeded_train = y_train[succeeded_synth_gt_train]
    X_raw_succeeded_train = X_raw_train[succeeded_synth_gt_train, :]

    # train the regressor
    print("Training the regressor...")
    # train_regressor(X_succeeded_train, y_succeeded_train)
    display_results_regressor(X_succeeded_test, y_succeeded_test, X_raw_succeeded_test, ["WorstLatency_hls", "IntervalMax_hls", "FF_hls", "LUT_hls", "BRAM_18K_hls", "DSP_hls"])

    print("Testing the models in union...")
    model_classifier = load_model("model_class_1/KERAS_check_best_model.h5")

    # predict the classes of the test dataset, convert to binary 1/0
    class_pred = model_classifier.predict(X_test_classifier)
    class_binary = np.where(class_pred > 1, 1, 0)
    print(class_binary)
    print(X_test.shape)

    # insert the (wrong) binary classification to the middle of the input data
    X_test_with_predicted_class = np.copy(X_test)
    X_test_with_predicted_class[:, 4] = class_binary[0,:]

    # get only the successes for prediction
    succeeded_idx = np.nonzero(class_binary)[0]
    print(succeeded_idx)
    X_test_only_success = X_test_with_predicted_class[succeeded_idx]

    model_regressor = load_model('model_reg_1/KERAS_check_best_model.h5')

    # Predict the regression outputs for X_test
    y_pred_without_class = model_regressor.predict(X_test_only_success)

    # Add back the classification as a prediction index (does not yet need to have the data)
    y_pred_with_class = np.append(y_pred_without_class, np.full((y_pred_without_class.shape[0], 1), 1), axis=1)

    # Restore the indices that fail classification
    y_pred_reshaped = np.empty((class_binary.shape[0], y_pred_with_class.shape[1]))
    y_pred_reshaped[succeeded_idx] = y_pred_with_class
    y_pred = np.where(class_binary > 0, y_pred_reshaped, -1)

    # set the classification predictions as the 0/1 binary for error calc
    y_pred[:, 6] = class_binary[0,:]
    y_test_total = np.append(y_test, np.reshape(y_test_classifier, (y_test_classifier.shape[0], 1)), axis = 1)

    print(y_pred.shape)
    print(y_pred)

    # Calculate metrics
    mae = mean_absolute_error(y_test_total, y_pred)
    mse = mean_squared_error(y_test_total, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_total, y_pred)

    # now set classification predictions to be the hinge classification raw outputs
    y_pred[:, 6] = class_pred[:,0]

    print(f'Mean Absolute Error (MAE): {mae}')
    print(f'Mean Squared Error (MSE): {mse}')
    print(f'Root Mean Squared Error (RMSE): {rmse}')
    print(f'R-squared (R2 Score): {r2}')

    plot_results("both", False, y_test_total, y_pred, X_raw_test_classifier, ["WorstLatency_hls", "IntervalMax_hls", "FF_hls", "LUT_hls", "BRAM_18K_hls","DSP_hls","hls_synth_success"])

    



    