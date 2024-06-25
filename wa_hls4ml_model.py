import torch
from torch import nn
import collections
import os
import sys

def load_model(model_dir):
    model = torch.load(model_dir+"/model.pth")
    model.eval()
    model.load_state_dict(torch.load(model_dir+"/model_weights.pth"))

    return model

def save_model(model, model_dir):

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model.eval()
    torch.save(model, model_dir+"/model.pth")
    torch.save(model.state_dict(), model_dir+"/model_weights.pth")

class ClassificationNet(nn.Module):
    def __init__(self):
        super().__init__()

        modules = collections.OrderedDict()

        modules['fc1'] = nn.Linear(in_features=6, out_features=128)
        modules['elu1'] = nn.ELU()
        modules['fc2'] = nn.Linear(in_features=128, out_features=2048)
        modules['relu1'] = nn.LeakyReLU()
        modules['dropout1'] = nn.Dropout(0.4)
        modules['fc3'] = nn.Linear(in_features=2048, out_features=4096)
        modules['relu2'] = nn.LeakyReLU()
        modules['dropout2'] = nn.Dropout(0.6)
        modules['fc4'] = nn.Linear(in_features=4096, out_features=2048)
        modules['relu3'] = nn.LeakyReLU()
        modules['dropout3'] = nn.Dropout(0.4)
        modules['fc5'] = nn.Linear(in_features=2048, out_features=128)
        modules['elu2'] = nn.ELU()
        modules['dropout4'] = nn.Dropout(0.2)
        modules['output'] = nn.Linear(in_features=128, out_features=1)
        modules['sigmoid'] = nn.Sigmoid()

        self.layers = nn.Sequential(modules)


    def forward(self, x):
        '''Forward pass'''
        return self.layers(x)

def create_model_classification():

    model = ClassificationNet().to('cpu')
    return model

class RegressionNet(nn.Module):
    def __init__(self):
        super().__init__()

        modules = collections.OrderedDict()

        modules['fc1'] =  nn.Linear(in_features=6, out_features=64)
        modules['elu1'] =  nn.ELU()
        modules['fc2'] =  nn.Linear(in_features=64, out_features=1024)
        modules['relu1'] =  nn.LeakyReLU()
        modules['dropout1'] =  nn.Dropout(0.2)
        modules['fc3'] =  nn.Linear(in_features=1024, out_features=1024)
        modules['relu2'] =  nn.LeakyReLU()
        modules['dropout2'] =  nn.Dropout(0.3)
        modules['fc4'] =  nn.Linear(in_features=1024, out_features=1024)
        modules['relu3'] =  nn.LeakyReLU()
        modules['dropout3'] =  nn.Dropout(0.2)
        modules['fc5'] =  nn.Linear(in_features=1024, out_features=64)
        modules['elu2'] =  nn.ELU()
        modules['output'] =  nn.Linear(in_features=64, out_features=1)  

        self.layers = nn.Sequential(modules)

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.0)

        self.layers.apply(init_weights)

    def forward(self, x):
        '''Forward pass'''
        return self.layers(x)


def create_model_regression_single_feature():
    model = RegressionNet().to('cpu')
    return model


# create initial graph embedding of node and edge states
# def gnn_set_initial_states():
#     def node_sets_fn(node_set, node_set_name):
#         if node_set_name == 'nn_layer':
#             # embedding for the nodes variable
#             embedded = tf.keras.layers.Embedding(1, 4)(node_set['nodes'])            
#             return Dense(64, activation='leaky_relu')(embedded)
#         else:
#             print("Unrecognized node feature")
#             sys.exit(1)
    
#     def edge_sets_fn(edge_set, edge_set_name):
#         if edge_set_name == 'feedforward':
            
#             # get all edge features
#             activation_1 = tf.keras.layers.Embedding(1,4)((edge_set['activation'])[:, 0])
#             activation_2 = tf.keras.layers.Embedding(1,4)((edge_set['activation'])[:, 1])
#             activation_3 = tf.keras.layers.Embedding(1,4)((edge_set['activation'])[:, 2])

#             density = tf.keras.layers.Embedding(1,4)((edge_set['density']))
#             dropout = tf.keras.layers.Embedding(1,4)((edge_set['dropout']))

#             # concatenate edge variables into one embedding
#             #TODO: does this actually work? no clue
#             concatenated_embedding = keras.layers.Concatenate(axis=1)([activation_1, activation_2, activation_3, density, dropout])

#             # TODO: this feels like the wrong way to deal with this...
#             total_embedding = Dense(64, activation='leaky_relu')(concatenated_embedding)
#             return total_embedding
#         else:
#             print("Unrecognized edge feature")
#             sys.exit(1)
    
#     return tfgnn.keras.layers.MapFeatures(node_sets_fn=node_sets_fn, edge_sets_fn=edge_sets_fn, name='graph_embedding')

# def gnn_feed(graph):
#     ''' Feed the graph through message-passing layers '''

#     # GNN hidden message-passing layers
#     graph = mt_albis.MtAlbisGraphUpdate(units=128, message_dim=128, receiver_tag=tfgnn.TARGET, kernel_initializer='lecun_uniform')(graph)
#     graph = mt_albis.MtAlbisGraphUpdate(units=1024, message_dim=128, receiver_tag=tfgnn.TARGET, kernel_initializer='lecun_uniform')(graph)
#     graph = mt_albis.MtAlbisGraphUpdate(units=128, message_dim=128, receiver_tag=tfgnn.TARGET, kernel_initializer='lecun_uniform')(graph)

#     return graph

# def create_model_gnn():
#     ''' Use a GNN framework to train a regression model '''

#     # use the schema to create an input layer of the right format, to anticipate data
#     graph_schema = tfgnn.read_schema("graph_schema.pbtxt")
#     graph_tensor_spec = tfgnn.create_graph_spec_from_schema_pb(graph_schema)

#     input_graph = keras.layers.Input(type_spec=graph_tensor_spec)

#     # perform initial embedding
#     graph_embedding = gnn_set_initial_states()
#     graph = graph_embedding(input_graph)

#     # the actual GNN happens here
#     hidden_graph = gnn_feed(graph)

#     # aggregate all the node and context information into one layer
#     hidden_state = tfgnn.keras.layers.Pool(tfgnn.CONTEXT, "mean", node_set_name="nn_layer")(hidden_graph)
#     hidden_state = keras.layers.Dense(units=128, activation='leaky_relu'] =  name='final_hidden_layer')(hidden_state)

#     # finally consolidate to a predictor head to give a regression output
#     predictor_head = keras.layers.Dense(units=1, name='predictor_head')(hidden_state)
        
#     model = keras.Model(inputs=input_graph, outputs=predictor_head)
#     print(model.summary())

#     return model