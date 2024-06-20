from tensorflow import keras

from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization, Dropout
from keras.regularizers import l1

import tensorflow_gnn as tfgnn
from tensorflow_gnn.models import mt_albis

import sys


def create_model_classification():
    model = Sequential()
    # d_in	d_out prec rf strategy (one-hot encoded)
    model.add(Dense(128, input_shape=(6,), name='fc1', kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.00001)))
    model.add(Activation(activation='elu', name='elu'))

    model.add(Dense(2048, name='fc2', kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.00001)))
    model.add(Activation(activation='leaky_relu', name='relu1'))
    model.add(Dropout(0.3))

    model.add(Dense(4096, name='fc3', kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.000001)))
    model.add(Activation(activation='leaky_relu', name='relu2'))
    model.add(Dropout(0.5))

    model.add(Dense(2048, name='fc4', kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.000001))) #32
    model.add(Activation(activation='leaky_relu', name='relu3'))
    model.add(Dropout(0.3))

    model.add(Dense(128, name='fc6', kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.00001))) #32
    model.add(Activation(activation='elu', name='elu2'))

    #hls_synth_success
    model.add(Dense(1, name='output', kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.001)))
    return model

def create_model_regression_single_feature():
    model = Sequential()
    # d_in	d_out prec rf strategy (one-hot encoded)
    model.add(Dense(64, input_shape=(6,), name='fc1', kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.00001)))
    model.add(Activation(activation='leaky_relu', name='relu1'))
    #model.add(BatchNormalization())
    model.add(Dense(1024, name='fc2', kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.00001)))
    model.add(Activation(activation='leaky_relu', name='relu2'))
    #model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(4096, name='fc3', kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.00001)))
    model.add(Activation(activation='elu', name='elu1'))
    #model.add(BatchNormalization())
    model.add(Dropout(0.7))
    model.add(Dense(1024, name='fc4', kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.00001))) #32
    model.add(Activation(activation='leaky_relu', name='relu3'))
    #model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(64, name='fc6', kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.00001))) #32
    model.add(Activation(activation='elu', name='elu2'))
    #hls_synth_success, WorstLatency_hls, IntervalMax_hls, FF_hls, LUT_hls, BRAM_18K_hls, DSP_hls
    model.add(Dense(1, name='output', kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.001)))
    return model


# create initial graph embedding of node and edge states
def gnn_set_initial_states():
    def node_sets_fn(node_set, node_set_name):
        if node_set_name == 'nn_layer':
            # embedding for the nodes variable
            return Dense(64, activation='leaky_relu')(node_set['nodes'])
        else:
            print("Unrecognized node feature")
            sys.exit(1)
    
    def edge_sets_fn(edge_set, edge_set_name):
        if edge_set_name == 'feedforward':
            
            # get all edge features
            activation_1 = (edge_set['activation'])[:, 0]
            activation_2 = (edge_set['activation'])[:, 1]
            activation_3 = (edge_set['activation'])[:, 2]

            density = edge_set['density']
            dropout = edge_set['dropout']

            # concatenate edge variables into one embedding
            #TODO: does this actually work? no clue
            concatenated_embedding = keras.layers.Concatenate(axis=1)([activation_1, activation_2, activation_3, density, dropout])

            # TODO: this feels like the wrong way to deal with this...
            total_embedding = Dense(64, activation='leaky_relu')(concatenated_embedding)
            return total_embedding
        else:
            print("Unrecognized edge feature")
            sys.exit(1)
    
    return keras.layers.MapFeatures(node_sets_fn=node_sets_fn, edge_sets_fn=edge_sets_fn, name='graph_embedding')

def gnn_feed(graph):
    ''' Feed the graph through message-passing layers '''

    # GNN hidden message-passing layers
    graph = mt_albis.MtAlbisGraphUpdate(128, kernel_initializer='lecun_uniform')(graph)
    graph = mt_albis.MtAlbisGraphUpdate(1024, kernel_initializer='lecun_uniform')(graph)
    graph = mt_albis.MtAlbisGraphUpdate(128, kernel_initializer='lecun_uniform')(graph)

    return graph

def create_model_gnn(X_train, y_train, output_features, folder_name):
    ''' Use a GNN framework to train a regression model '''

    # use the schema to create an input layer of the right format, to anticipate data
    graph_schema = tfgnn.read_schema("graph_schema.pbtxt")
    graph_tensor_spec = tfgnn.create_graph_spec_from_schema_pb(graph_schema)

    input_graph = keras.layers.Input(type_spec=graph_tensor_spec)

    # perform initial embedding
    graph_embedding = gnn_set_initial_states()
    graph = graph_embedding(input_graph)

    # the actual GNN happens here
    hidden_graph = gnn_feed(graph)

    # aggregate all the node and context information into one layer
    hidden_state = tfgnn.keras.layers.Pool(tfgnn.CONTEXT, "mean", node_set_name="nn_layer")(hidden_graph)
    hidden_state = keras.layers.Dense(units=128, activation='leaky_relu', name='final_hidden_layer')(hidden_state)

    # finally consolidate to a predictor head to give a regression output
    predictor_head = keras.layers.Dense(units=1, name='predictor_head')(hidden_state)
        
    model = keras.Model(inputs=input_graph, outputs=predictor_head)
    print(model.summary())

    return model