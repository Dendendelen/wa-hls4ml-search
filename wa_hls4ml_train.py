import os
import sys
import argparse
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn.model_selection
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
import tensorflow_gnn as tfgnn

# This seems like you should just be able to import straight from tensorflow, but that messes up syntax checking
from tensorflow import keras

from keras.optimizers import Adam, AdamW
from keras.models import load_model

import keras.backend as K
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from wa_hls4ml_plotly import plot_results
import wa_hls4ml_model
import wa_hls4ml_data_plot

from callbacks import all_callbacks

def stack_dict(inputs, fun=tf.stack):
    values = []
    for key in sorted(inputs.keys()):
        values.append(tf.cast(inputs[key], tf.float32))

    return fun(values, axis=-1)


# current I/O:
#   inputs: d_in,	d_out, prec, rf, strategy
#   outputs: TargetClockPeriod_hls,	WorstLatency_hls, IntervalMax_hls, FF_hls, LUT_hls, BRAM_18K_hls, DSP_hls, hls_synth_success

def preprocess_data_from_csv(csv_file, input_features, output_features, _binary_feature_names, _numeric_feature_names, _categorical_feature_names, special_feature_names):
    ''' Extract data from a CSV, and preprocess that data '''

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

    special_data = []

    # Step 8: Extract special features
    # these are features which we do not want to process at all
    for i in range (preprocessed_data[0].shape[0]):
        special_datapoint = []
        for name in special_feature_names:
            special_feature = df[name][i]
            special_datapoint.append(special_feature)
        special_data.append(special_datapoint)

    return np.nan_to_num(preprocessed_data[0], nan=-1), np.nan_to_num(preprocessed_data[1], nan=-1), np.nan_to_num(preprocessed_data[2], nan=-1), special_data


def preprocess_features(data, binary_feature_names, numeric_feature_names, categorical_feature_names,processing_input=True):
    ''' Preprocess features '''
    preprocessed = []

    # Step 3: Process numeric features
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

    # Step 4: Process binary features
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


def parse_json_string(json):
    ''' Parse the model information out of a JSON string ''' 
    #TODO: implement
    return json


def create_graph_tensor(input_values, input_json):
    ''' Turn the data into the form of a GraphTensor to allow for GNN use ''' 

    #TODO: check that these match the actual order of parsing
    precision = input_values[0]
    rf = input_values[1]
    strategy = input_values[2:]

    #TODO: parse json into distinct nodes and edges
    nodes_count, source, target, activation, density, dropout = parse_json_string(input_json)

    # create GraphTensor from these lists
    nn_layer = tfgnn.NodeSet.from_fields(features={'nodes':nodes_count}, sizes=tf.shape(nodes_count))
    nn_adjacency = tfgnn.Adjacency.from_indices(source=('nn_layer', source), target=('nn_layer', target))
    feedforward = tfgnn.EdgeSet.from_fields(features={'activation': activation, 'density': density, 'dropout': dropout}, sizes= (len(source)), adjacency = nn_adjacency)
    context = tfgnn.Context.from_fields(features={'rf': [rf], 'precision': [precision], 'strategy': [strategy]})
    
    return tfgnn.GraphTensor.from_pieces(node_sets={'nn_layer': nn_layer}, edge_sets={'feedforward': feedforward}, context=context)


def preprocess_data(is_graph = False):
    ''' Preprocess the data '''

    input_features = ["d_in", "d_out", "prec", "rf", "strategy"]
    output_features = ["WorstLatency_hls", "IntervalMax_hls", "FF_hls", "LUT_hls", "BRAM_18K_hls",
                       "DSP_hls", "hls_synth_success"]
    binary_feature_names = ['hls_synth_success']
    numeric_feature_names = ["d_in", "d_out", "prec", "rf", "WorstLatency_hls", "IntervalMax_hls", "FF_hls", "LUT_hls",
                             "BRAM_18K_hls", "DSP_hls"]
    categorical_feature_names = ["strategy"]
    # special_feature_names = ["json"]
    special_feature_names = []

    _X, y, X_raw, special_data = preprocess_data_from_csv('results/results_format_test.csv', input_features, output_features,
                             binary_feature_names, numeric_feature_names,
                             categorical_feature_names, special_feature_names)

    if (is_graph):
        i = 0
        graph_tensor_list = []
        for datapoint in special_data:
            # tensorize this data into the special graph-tensor format
            graph_tensor = create_graph_tensor(_X[i], datapoint)
            graph_tensor_list.append(graph_tensor)
            i += 1

        graph_schema = tfgnn.read_schema("graph_schema.pbtxt")
        graph_tensor_spec = tfgnn.create_graph_spec_from_schema_pb(graph_schema)

        if not graph_tensor_spec.is_compatible_with(graph_tensor_list[0]):
            print("Error: this tensor does not fit the schema")
            sys.exit(1)

        X = graph_tensor_list
        #TODO: check if works
    else:
        X = _X

    # Split the data 70 - 20 - 10 train test val
    # Train and test
    print(X.shape, y.shape)
    print("X Data: ",input_features)
    print(X)
    print("Y Data: ",output_features)
    print(y[0])
    return sklearn.model_selection.train_test_split(X, y, X_raw,  test_size=0.2, random_state=42, shuffle=True)


def train_classifier(X_train, y_train, folder_name):
    ''' Train classification model '''

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
        outputDir=folder_name+'/classification',
    )
    history = model.fit(
        X_train,
        y_train,
        batch_size=256,
        epochs=250,
        validation_split=0.125,
        shuffle=True,
        callbacks=callbacks.callbacks,
    )
    wa_hls4ml_data_plot.plot_loss('classifier', history, folder_name)
    

def train_regressor(X_train, y_train, output_features, graph, folder_name):
    ''' Train regression models for all features '''

    i = 0
    for feature in output_features:

        # Train regression model to target individual output feature
        print("Training " + feature + "...")
        y_train_feature = y_train[:, i]

        if not graph:
            model = wa_hls4ml_model.create_model_regression_single_feature()
        else:
            model = wa_hls4ml_model.create_model_gnn()

        adam = AdamW(learning_rate=0.005)
        model.compile(optimizer=adam, loss=tf.keras.losses.Huber(), metrics=['mse', 'mae'])
        callbacks = all_callbacks(
            stop_patience=1000,
            lr_factor=0.5,
            lr_patience=8,
            lr_epsilon=0.0001,
            lr_cooldown=3,
            lr_minimum=0.0000000001,
            outputDir=folder_name+'/regression_'+feature,
        )
        history = model.fit(
            X_train,
            y_train_feature,
            batch_size=128,
            epochs=200,
            validation_split=0.125,
            shuffle=True,
            callbacks=callbacks.callbacks,
        )
        wa_hls4ml_data_plot.plot_loss('regression_'+feature,history, folder_name)

        i += 1


def calculate_metrics(y_test, y_pred):
    ''' Calculate out MAE, MSE, RMSE, and R^2 '''

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f'Mean Absolute Error (MAE): {mae}')
    print(f'Mean Squared Error (MSE): {mse}')
    print(f'Root Mean Squared Error (RMSE): {rmse}')
    print(f'R-squared (R2 Score): {r2}')


def display_results_classifier(X_test, X_raw_test, y_test, output_features, folder_name):
    ''' Display the results of the classification model '''

    model = load_model(folder_name+'/classification/KERAS_check_best_model.h5')

    # Predict the output for X_test
    y_pred = model.predict(X_test)

    # Calculate metrics
    calculate_metrics(y_test, y_pred)

    # Calculate confusion matrix
    y_pred_binary = np.where(y_pred > 1, 1, 0)
    confusion = confusion_matrix(y_pred_binary, y_test)
    print('Confusion matrix:')
    print(confusion)


    # plot our classification results
    y_test_2d = np.reshape(y_test, (y_test.shape[0], 1))
    plot_results("classifier", False, y_test_2d, y_pred, X_raw_test, output_features, folder_name)



def display_results_regressor(X_test, X_raw_test, y_test, output_features, folder_name):
    ''' Display the results of the regression models '''

    y_pred = np.empty(y_test.shape)

    i = 0
    for feature in output_features:
        model = load_model(folder_name+'/regression_'+ feature +'/KERAS_check_best_model.h5')

        # Predict the output of this specific feature for X_test
        y_pred_part = model.predict(X_test)
        print(y_pred_part.shape)

        # Consolidate feature predictions
        y_pred[:, i] = y_pred_part[:, 0]

        i += 1

    # Calculate metrics
    calculate_metrics(y_test, y_pred)

    # plot our regression results
    plot_results("regression_all", False, y_test, y_pred, X_raw_test, output_features, folder_name)


def test_regression_classification_union(X_test, X_raw_test, y_test, features_without_classification, feature_classification_task, folder_name):
    '''Test the effectiveness of the whole model, first doing classification, and using that result to help with regression'''

    features_with_classification = features_without_classification + feature_classification_task

    model_classifier = load_model(folder_name+"/classification/KERAS_check_best_model.h5")

    # predict the classes of the test dataset, then convert to binary 1/0
    class_pred = model_classifier.predict(X_test)
    class_binary = np.where(class_pred > 0, 1, 0)
    print("Binary classification created, shape:")
    print(class_binary.shape)

    # get only the predicted successes for regression prediction
    succeeded_idx = np.nonzero(class_binary)[0]
    print("Indices of success created, shape:")
    print(succeeded_idx.shape)
    X_test_only_success = X_test[succeeded_idx]

    y_regression_pred = np.empty((X_test_only_success.shape[0], 6))

    i = 0
    for feature in features_without_classification:
        model_regressor = load_model(folder_name+'/regression_' + feature + '/KERAS_check_best_model.h5')
        y_regression_pred_slice = model_regressor.predict(X_test_only_success)
        y_regression_pred[:, i] = y_regression_pred_slice[:, 0]
        i += 1

    # Add back the classification as a prediction index (does not yet need to have the data)
    y_regression_pred_with_classification_column = np.append(y_regression_pred, np.full((y_regression_pred.shape[0], 1), 1), axis=1)

    # Restore the indices that fail classification
    y_regression_pred_reshaped = np.empty((class_binary.shape[0], y_regression_pred_with_classification_column.shape[1]))
    y_regression_pred_reshaped[succeeded_idx] = y_regression_pred_with_classification_column

    y_pred = np.where(class_binary > 0, y_regression_pred_reshaped, -1)

    # set the classification predictions as the 0/1 binary for error calc
    y_pred[:, -1] = class_binary[:, 0]

    print(y_pred.shape)

    # Calculate metrics
    calculate_metrics(y_test, y_pred)

    # Generate histograms for all
    wa_hls4ml_data_plot.plot_histograms(y_pred, y_test, features_with_classification, folder_name)

    # now set classification predictions to be the hinge classification raw outputs
    y_pred[:, 6] = class_pred[:,0]

    # use plotly to create result graph
    plot_results("both", False, y_test, y_pred, X_raw_test, features_with_classification, folder_name)
    

def perform_train_and_test(train, test, regression, classification, skip_intermediates, is_graph, folder_name = "model_1"):

    features_without_classification = ["WorstLatency_hls", "IntervalMax_hls", "FF_hls", "LUT_hls", "BRAM_18K_hls", "DSP_hls"]
    feature_classification_task = ["hls_synth_success"]

    # get raw data out
    X_train, X_test, y_train, y_test, X_raw_train, X_raw_test = preprocess_data(is_graph)

    # get just the classification task as its own variable
    y_train_classifier = y_train[:, -1]
    y_test_classifier = y_test[:, -1]

    # train the classifier
    if train and classification:
        print("Training the classifier...")
        train_classifier(X_train, y_train_classifier, folder_name)
    if test and classification and not skip_display_intermediate:
        display_results_classifier(X_test, X_raw_test, y_test_classifier, feature_classification_task, folder_name)

    # This is all we do if we are doing classification alone
    if not regression:
        return

    # find which synth has succeeded in groundtruth
    succeeded_synth_gt_test = np.nonzero(y_test_classifier)
    succeeded_synth_gt_train = np.nonzero(y_train_classifier)

    # only train regressor on successes
    X_succeeded_train = X_train[succeeded_synth_gt_train]
    y_succeeded_train = (y_train[succeeded_synth_gt_train])[:, :-1]
    X_raw_succeeded_train = X_raw_train[succeeded_synth_gt_train]

    # only test regressor alone on successes
    X_succeeded_test = X_test[succeeded_synth_gt_test]
    y_succeeded_test = (y_test[succeeded_synth_gt_test])[:, :-1]
    X_raw_succeeded_test = X_raw_test[succeeded_synth_gt_test]

    # train the regressor
    if train and regression:
        print("Training the regressor...")
        train_regressor(X_succeeded_train, y_succeeded_train, features_without_classification, is_graph, folder_name)
    if test and regression and not skip_intermediates:
        display_results_regressor(X_succeeded_test, X_raw_succeeded_test, y_succeeded_test, features_without_classification, folder_name)

    # if we are not doing both regression and classification, we are done
    if not regression or not classification or not test:
        return
    
    print("Testing the models in union...")
    test_regression_classification_union(X_test, X_raw_test, y_test, features_without_classification, feature_classification_task, folder_name)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog='wa-hls4ml', description='Train or test model for wa-hls4ml', add_help=True)

    parser.add_argument('--train', action='store_true', help='Train a new surrogate model from the data')
    parser.add_argument('--test', action='store_true', help='Test existing models')

    parser.add_argument('-c', '--classification', action='store_true', help='Train/test the classifier')
    parser.add_argument('-r','--regression', action='store_true', help='Train/test the regressor')

    parser.add_argument('-g','--gnn', action='store_true', help='Use a graph neural network to model the layers of the ml model')

    parser.add_argument('-f', '--folder', action='store', help='Set the folder you want the model outputs to be created within')

    args = parser.parse_args()
    args_dict = vars(args)

    # allow for GPU use if availible
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

    train = args_dict['train']
    test = args_dict['test']

    # if neither flag is assigned, assume both
    if not train and not test:
        train = True
        test = True

    classification = args_dict['classification']
    regression= args_dict['regression']

    skip_display_intermediate = False

    # if neither flag is assigned, assume both, but that we do not care about the intermediate classification/regression results
    if not classification and not regression:
        classification = True
        regression = True
        skip_display_intermediate = True

    # whether or not to use the GNN
    is_graph = args_dict['gnn']

    # folder to store all the outputs into
    folder = args_dict['folder']

    # perform the testing and training depending on arguments
    print("Beginning...")
    perform_train_and_test(train, test, regression, classification, skip_display_intermediate, is_graph, folder)
    print("Done")
    



    