import pandas as pd
import numpy as np
import torch
import sklearn

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
            tensorized_val = torch.tensor(data[numeric_feature_names].astype('float32').values)
            mean, stdev = tensorized_val.mean(dim=0).broadcast_to(tensorized_val.shape), tensorized_val.std(dim=0).broadcast_to(tensorized_val.shape)
            numeric_normalized = (tensorized_val-mean)/stdev
        else:
            numeric_normalized = torch.tensor(data[numeric_feature_names].values.astype('float32'))
        preprocessed.append(numeric_normalized)

    # Step 4: Process binary features
    if binary_feature_names:
        for name in binary_feature_names:
            value = data[name].astype(bool).astype('float32')
            value = torch.tensor(value).unsqueeze(1)
            # value = torch.tensor(2*value - 1).unsqueeze(1)


            # value = tf.reshape(value, [-1, 1])
            preprocessed.append(value)

    # Step 5: Process categorical features
    if categorical_feature_names:
        for name in categorical_feature_names:
            vocab = sorted(set(data[name][1:])) #Exclude header
            if type(vocab[0]) is str:
                # change strings to integers
                i = 0
                for word in vocab:
                    data[name] = data[name].replace(word, i)
                    i += 1

            numbered_data = torch.tensor(data[name])

            one_hot = torch.zeros(numbered_data.shape[0], len(vocab))
            one_hot.scatter_(1, numbered_data.unsqueeze(1), 1.0)

            print(data[name].shape)
            print(one_hot)
            preprocessed.append(one_hot)

    # Step 6: Concatenate all processed features

    preprocessed_data = torch.cat(preprocessed, dim=1)
    print(preprocessed_data)

    return preprocessed_data


def parse_json_string(json):
    ''' Parse the model information out of a JSON string ''' 
    #TODO: implement

    nodes_count = np.asarray([json[0], json[1]])

    source = np.zeros((1,)).astype('int64')
    target = np.ones((1,)).astype('int64')

    activation = np.zeros((1,3)).astype('int64')
    # activation[0, 0] = 1

    density = np.ones((1,)).astype('float32')
    dropout = np.zeros((1,)).astype('float32')

    return nodes_count, source, target, activation, density, dropout


# def create_graph_tensor(input_values, input_json):
#     ''' Turn the data into the form of a GraphTensor to allow for GNN use ''' 

#     # ------------------ testing ---------------

#     input_json = input_values[:2]

#     input_values_2 = np.asarray(input_values[2:]).astype('int64')

#     # --------------------testing -------------- TODO:remove


#     #TODO: check that these match the actual order of parsing

#     precision = np.asarray(input_values_2).astype('int64')[0]
#     rf = np.asarray(input_values_2).astype('int64')[1]
#     strategy = np.asarray(input_values_2).astype('int64')[2:]

#     #TODO: parse json into distinct nodes and edges
#     nodes_count, source, target, activation, density, dropout = parse_json_string(input_json)

#     # create GraphTensor from these lists
#     nn_layer = tfgnn.NodeSet.from_fields(features={'nodes':tf.cast(nodes_count, tf.int64)}, sizes=(len(nodes_count),))
#     nn_adjacency = tfgnn.Adjacency.from_indices(source=('nn_layer', source), target=('nn_layer', target))
#     feedforward = tfgnn.EdgeSet.from_fields(features={'activation': activation, 'density': density, 'dropout': dropout}, sizes= (1,), adjacency = nn_adjacency)
#     context = tfgnn.Context.from_fields(features={'rf': [tf.cast(rf, tf.int64)], 'precision': [tf.cast(precision, tf.int64)], 'strategy': [tf.cast(strategy, tf.int64)]})
    
#     return tfgnn.GraphTensor.from_pieces(node_sets={'nn_layer': nn_layer}, edge_sets={'feedforward': feedforward}, context=context)    

# def serialize_graph_data(filename, graph_list):
#     # store to a data file
#     record_file = filename+".tfrecord"
#     with tf.io.TFRecordWriter(record_file) as writer:
#         for graph in graph_list:
#             example = tfgnn.write_example(graph)
#             writer.write(example.SerializeToString())

    
# def unserialize_graph_data(filename, graph_tensor_spec):
#     dataset = tf.data.TFRecordDataset(filenames=[filename+".tfrecord"])
#     dataset = dataset.map(
#         lambda serialized: tfgnn.parse_single_example(graph_tensor_spec, serialized))
#     return dataset


def preprocess_data(is_graph = False, is_already_serialized = True):
    ''' Preprocess the data '''

    input_features = ["d_in", "d_out", "prec", "rf", "strategy"]
    output_features = ["WorstLatency_hls", "IntervalMax_hls", "FF_hls", "LUT_hls", "BRAM_18K_hls", "DSP_hls", "hls_synth_success"]
    binary_feature_names = ['hls_synth_success']
    numeric_feature_names = ["d_in", "d_out", "prec", "rf", "WorstLatency_hls", "IntervalMax_hls", "FF_hls", "LUT_hls",
                             "BRAM_18K_hls", "DSP_hls"]
    categorical_feature_names = ["strategy"]
    # special_feature_names = ["json"]
    special_feature_names = ["model_name"]

    _X, y, X_raw, special_data = preprocess_data_from_csv('results/results_format_test.csv', input_features, output_features,
                             binary_feature_names, numeric_feature_names,
                             categorical_feature_names, special_feature_names)

    if (is_graph and not is_already_serialized):
        i = 0
        graph_tensor_list = []

        for datapoint in special_data:
            # tensorize this data into the special graph-tensor format
            graph_tensor = create_graph_tensor(X_raw[i], datapoint)
            graph_tensor_list.append(graph_tensor)
            i += 1
            if i % 100 == 0:
                print(i)

        # confirm_new_spec = tfgnn.create_graph_spec_from_schema_pb(tfgnn.create_schema_pb_from_graph_spec(graph_tensor_list[0]))
        # if not graph_tensor_spec.is_compatible_with(confirm_new_spec):
        #     print("Error: this tensor does not fit the schema")
        #     sys.exit(1)

        X = graph_tensor_list
    else:
        X = _X
        print(X.shape, y.shape)

    # Split the data 70 - 20 - 10 train test val
    # Train and test
    print("X Data: ",input_features)
    print(X)
    print("Y Data: ",output_features)
    print(y[0])

    X_train, X_test, y_train, y_test, X_raw_train, X_raw_test = sklearn.model_selection.train_test_split(X, y, X_raw,  test_size=0.2, random_state=42, shuffle=True)


    # if (is_graph):
    #     graph_schema = tfgnn.read_schema("graph_schema.pbtxt")
    #     graph_tensor_spec = tfgnn.create_graph_spec_from_schema_pb(graph_schema)

    #     if not is_already_serialized:
    #         serialize_graph_data("train_for_gnn", X_train)
    #     train_dataset = unserialize_graph_data("train_for_gnn", graph_tensor_spec)
    #     train_dataset = next(iter(train_dataset.batch(len(X_train), drop_remainder=True)))

    #     if not is_already_serialized:
    #         serialize_graph_data("test_for_gnn", X_test)
    #     test_dataset = unserialize_graph_data("test_for_gnn", graph_tensor_spec)
    #     test_dataset = next(iter(test_dataset.batch(len(X_test), drop_remainder=True)))

    #     X_train = train_dataset.merge_batch_to_components()
    #     X_test = test_dataset.merge_batch_to_components()

    return X_train, X_test, y_train, y_test, X_raw_train, X_raw_test
