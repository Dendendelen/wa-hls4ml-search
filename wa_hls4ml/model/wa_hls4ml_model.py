import torch
from torch import nn

import numpy as np

from torch_geometric import nn as gnn
from torch_geometric.utils import add_self_loops, degree

import collections
import os
import sys
import math

def load_model(model_dir):
    model = torch.load(model_dir+"/model.pth")
    model.eval()
    model.load_state_dict(torch.load(model_dir+"/model_weights.pth", map_location=torch.device("cpu")))

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

        modules['fc1'] = nn.Linear(in_features=7, out_features=128)
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


def create_model_classification(dev):

    model = ClassificationNet().to(cuda)
    return model

class RegressionNet(nn.Module):
    def __init__(self):
        super().__init__()

        modules = collections.OrderedDict()
        #TODO: check performance, these are double previous values
        modules['fc1'] =  nn.Linear(in_features=7, out_features=128)
        modules['elu1'] =  nn.ELU()
        modules['fc2'] =  nn.Linear(in_features=128, out_features=2048)
        modules['relu1'] =  nn.LeakyReLU()
        modules['dropout1'] =  nn.Dropout(0.2)
        modules['fc3'] =  nn.Linear(in_features=2048, out_features=2048)
        modules['relu2'] =  nn.LeakyReLU()
        modules['dropout2'] =  nn.Dropout(0.3)
        modules['fc4'] =  nn.Linear(in_features=2048, out_features=2048)
        modules['relu3'] =  nn.LeakyReLU()
        modules['dropout3'] =  nn.Dropout(0.2)
        modules['fc5'] =  nn.Linear(in_features=2048, out_features=128)
        modules['elu2'] =  nn.ELU()
        modules['output'] =  nn.Linear(in_features=128, out_features=1)  

        self.layers = nn.Sequential(modules)

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.0)

        self.layers.apply(init_weights)

    def forward(self, x):
        '''Forward pass'''
        return self.layers(x)
    

def create_model_regression_single_feature(dev):
    model = RegressionNet().to(dev)
    return model


class GraphNet(nn.Module):
    def __init__(self, num_global_features = 5, classification = True, dev = "cpu"):
        super().__init__()

        self.global_features = num_global_features
        self.dev = dev

        preprocessing_node = collections.OrderedDict()
        preprocessing_node['preproc_nodes_fc1'] = nn.Linear(in_features=1, out_features=16)
        preprocessing_node['preproc_nodes_relu1'] = nn.ELU()
        preprocessing_node['preproc_nodes_dropout'] = nn.Dropout(0.2)
        preprocessing_node['preproc_nodes_batchnorm'] = nn.BatchNorm1d(16)
        self.preprocessing_node_nn = nn.Sequential(preprocessing_node)

        preprocessing_edge = collections.OrderedDict()
        preprocessing_edge['preproc_edges_fc1'] = nn.Linear(in_features=1, out_features=2)
        preprocessing_edge['preproc_edges_relu1'] = nn.ELU()
        preprocessing_edge['preproc_edges_dropout'] = nn.Dropout(0.9)
        preprocessing_edge['preproc_edges_batchnorm'] = nn.BatchNorm1d(2)
        self.preprocessing_edge_nn = nn.Sequential(preprocessing_edge)

        preprocessing_global = collections.OrderedDict()
        preprocessing_global['preproc_glob_fc1'] = nn.Linear(in_features=num_global_features, out_features=256)
        preprocessing_global['preproc_glob_relu1'] = nn.ELU()
        preprocessing_global['preproc_glob_dropout'] = nn.Dropout(0.05)
        # preprocessing_global['preproc_glob_fc2'] = nn.Linear(in_features=128, out_features=2048)
        # preprocessing_global['preproc_glob_relu2'] = nn.ELU()
        # preprocessing_global['preproc_glob_dropout2'] = nn.Dropout(0.2)
        # preprocessing_global['preproc_glob_batchnorm'] = nn.BatchNorm1d(256)
        self.preprocessing_glob_nn = nn.Sequential(preprocessing_global)

        # self.gat_layer = gnn.GATv2Conv(in_channels=8, out_channels=1024, heads=4, edge_dim=2, add_self_loops=False, concat=False, dropout=0.2)
        self.gat_layer = gnn.GATv2Conv(in_channels=16, out_channels=128, heads=4, add_self_loops=False, concat=False, share_weights=False, dropout=0.3)

        # self.multiply = gnn.EquilibriumAggregation(in_channels=1024, out_channels=1024, num_layers=[512,512])
        self.multiply = gnn.MulAggregation()
        # self.multiply = gnn.SumAggregation()
        self.middle_message_layer = gnn.SimpleConv(aggr=self.multiply)
        self.middle_message_activation = nn.LeakyReLU()
        
        # self.gat_layer = gnn.GATv2Conv(in_channels=8, out_channels=1024, heads=1, edge_dim=2, add_self_loops=False, concat=False, dropout=0.2)
        self.gat_layer_2 = gnn.GATv2Conv(in_channels=128, out_channels=128, heads=4, add_self_loops=False, concat=False, dropout=0.3, edge_dim=4)
        # self.middle_message_layer = gnn.GeneralConv(in_channels=1024, out_channels=1024, in_edge_channels=6, directed_msg=False)

        self.multiply_2 = gnn.MulAggregation()
        self.middle_message_layer_2 = gnn.SimpleConv(aggr=self.multiply_2)
        self.middle_message_activation_2 = nn.LeakyReLU()
        
        self.message_norm = gnn.norm.GraphNorm(in_channels=128)

        edge_attention_embedding = collections.OrderedDict()
        edge_attention_embedding['edge_embed_fc1'] = nn.Linear(in_features = 6, out_features=128)
        edge_attention_embedding['edge_embed_elu1'] = nn.ELU()
        edge_attention_embedding['edge_embed_dropout'] = nn.Dropout(0.4)
        edge_attention_embedding['edge_embed_output'] = nn.Linear(in_features=128, out_features=128)
        self.edge_attention_embedding_layer = nn.Sequential(edge_attention_embedding)

        
        # self.middle_message_layer = gnn.GeneralConv(in_channels=8, out_channels=1024, in_edge_channels=2, directed_msg=False)
        # self.middle_message_layer = gnn.GeneralConv(in_channels=1024, out_channels=1024, directed_msg=False)

        # self.equilibrium = gnn.EquilibriumAggregation(in_channels=1024, out_channels=1024, num_layers=[1024,1024])
        # self.pooling_nodes = gnn.pool.global_add_pool

        
        # self.pooling_nodes = gnn.EquilibriumAggregation(in_channels=128, out_channels=128, num_layers=[1024], grad_iter=30, lamb=0.0001)
        self.pooling_nodes = gnn.MulAggregation()
        self.pooling_nodes_norm = nn.BatchNorm1d(128)
        
        self.pooling_edges = gnn.pool.global_add_pool
        self.pooling_edges_norm = nn.BatchNorm1d(128)

        final_pool = collections.OrderedDict()
        final_pool['pool_fc1'] = nn.Linear(in_features=(256+128+128), out_features=2048)
        # final_pool['pool_fc1'] = nn.Linear(in_features=(1024+1024+1024), out_features=2048)
        final_pool['pool_relu1'] = nn.ELU()
        final_pool['pool_dropout'] = nn.Dropout(0.2)
        final_pool['pool_fc2'] = nn.Linear(in_features=2048, out_features=2048)
        final_pool['pool_relu2'] = nn.ELU()
        final_pool['pool_dropout2'] = nn.Dropout(0.2)
        final_pool['pool_fc3'] = nn.Linear(in_features=2048, out_features=256)
        final_pool['pool_relu3'] = nn.ELU()
        final_pool['pool_output'] = nn.Linear(in_features=256, out_features=1)

        # if doing binary classification, we need a final sigmoid layer to turn logits into 0-1 range
        if classification:
            final_pool['pool_sigmoid'] = nn.Sigmoid()

        self.final_pooling_layer = nn.Sequential(final_pool)

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.0)

        self.preprocessing_node_nn.apply(init_weights)
        self.preprocessing_edge_nn.apply(init_weights)
        self.preprocessing_glob_nn.apply(init_weights)
        self.edge_attention_embedding_layer.apply(init_weights)
        self.final_pooling_layer.apply(init_weights)

        self.gat_layer.reset_parameters()
        self.gat_layer_2.reset_parameters()
        self.middle_message_layer.reset_parameters()
        self.pooling_nodes.reset_parameters()
        

    def preprocessing_layer(self, x, edge_index, edge_attr, y):
        ''' Perform a pre-convolution processing pass on the data, adding self-loops and passing features through individual neural nets '''

        edge_index, edge_attr = add_self_loops(edge_index, edge_attr=edge_attr, num_nodes=x.size(0), fill_value="add")
        
        # we would really like our global features to not be in one big list, let's reshape them
        num_total_glob_features = y.shape[0]
        batch_size = int(num_total_glob_features / self.global_features)
        y = torch.reshape(y, (batch_size, self.global_features))

        # np.save("dumpglob.npy", y.to("cpu").detach().numpy())
        # sys.exit(0)

        # the last of our global features is the number of edges in each graph. We should store that for later
        self.num_edges_per_graph = y[:, -1]

        # perform initial MLPs to get representations of our input features
        x = self.preprocessing_node_nn(x)
        edge_attr = self.preprocessing_edge_nn(edge_attr)
        y = self.preprocessing_glob_nn(y)

        return x, edge_index, edge_attr, y


    def message_passing_layer(self, x, edge_index, edge_attr, batch_vector):
        ''' Perform the actual meat of the GNN, a Graph Attention layer followed by a Graph Convolution layer (which incorporates attention as a feature) '''

        # x, _ = self.gat_layer(x=x, edge_index=edge_index, edge_attr=edge_attr, return_attention_weights=False)
        # out, edge_tuple = self.gat_layer(x=x, edge_index=edge_index, edge_attr=edge_attr, return_attention_weights=True)

        out, edge_tuple = self.gat_layer(x=x, edge_index=edge_index, return_attention_weights=True)
        # _, edge_attention = edge_tuple
        edge_index, edge_attention = edge_tuple

        out = self.middle_message_layer(x=out, edge_index=edge_index)
        out_1 = self.middle_message_activation(out)

        out, edge_tuple = self.gat_layer_2(x=out_1, edge_index=edge_index, edge_attr=edge_attention, return_attention_weights=True)
        edge_index, edge_attention = edge_tuple

        out = self.middle_message_layer_2(x=out, edge_index = edge_index)
        out_2 = self.middle_message_activation_2(out)

        # out, edge_tuple = self.gat_layer_2(x=out, edge_index=edge_index, return_attention_weights=True)
        # out, edge_tuple = self.gat_layer_2(x=out, edge_index=edge_index, edge_attr=edge_attention, return_attention_weights=True)

        # out = torch.cat((out_1, out_2), dim=1)
        out = out_2
        
        out = self.message_norm(out, batch=batch_vector)

        if torch.numel(edge_attention) == 1:
            edge_attention = edge_attention.unsqueeze(0)

        # add the attention weights as a feature to the edges themselves
        edge_total = torch.cat((edge_attr, edge_attention), dim=1)

        edge_total = self.edge_attention_embedding_layer(edge_total)

        
        # out = torch.cat((out, out_2), dim=1)
        # out = self.middle_message_layer(x=out, edge_index=edge_index, edge_attr=edge_total)
        # out = self.middle_message_layer(x=x, edge_index=edge_index, edge_attr=edge_attr)
        # edge_total = edge_attr

        return out, edge_total


    def pooling_layer(self, x, edge_index, edge_attr, y, batch_vector):
        ''' Perform pooling of the data, combining features for all nodes/all edges into a single respective feature, and then putting the combined lump through a neural net to produce final output '''
        
        # pool the nodes
        out_nodes = self.pooling_nodes(x, index=batch_vector.to(self.dev))
        out_nodes = self.pooling_nodes_norm(out_nodes)

        
        # out_nodes = self.pooling_nodes(x, index=batch_vector)

        # We would really like to pool edge features as well. Unfortunately, there isn't a built in way for it
        # Thus, we need to fabricate an edge batch vector
        edge_batch_vector = torch.empty((edge_attr.shape[0]), dtype=torch.int64)

        # get how many nodes are in each graph in the batch, since we need it to account for self loops
        num_nodes_per_graph = torch.bincount(batch_vector)

        index = 0
        for i in range(self.num_edges_per_graph.shape[0]):

            # we add the number of edges we started with to the number of nodes in our graph, since we add self-loops
            # NOTE: this assumes the incoming graph has no self-loops to begin with. If it does, account for that fact in the input data, since they will be added regardless
            num_edges = int(self.num_edges_per_graph[i].item()) + num_nodes_per_graph[i]

            # for every edge in this graph, we add one entry depicting which graph it belongs to
            for j in range(num_edges):
                # add the current graph we are on to the list
                edge_batch_vector[index] = i
                index += 1

        # pool the edges
        out_edges = self.pooling_edges(edge_attr, batch=edge_batch_vector.to(self.dev))
        out_edges = self.pooling_edges_norm(out_edges)

        # concatenate the pooled nodes, pooled edges, and global features into one large feature vector. Then, send that through a final NN for regression output
        pooled = torch.cat((out_nodes, out_edges, y), dim=1)
        out = self.final_pooling_layer(pooled)

        return out

    def forward(self, graph):
        '''Forward pass'''

        x = graph.x
        edge_index = graph.edge_index
        edge_attr = graph.edge_attr
        y = graph.y
        batch_vector = graph.batch

        x, edge_index, edge_attr, y = self.preprocessing_layer(x, edge_index, edge_attr, y)
        x, edge_attr = self.message_passing_layer(x, edge_index, edge_attr, batch_vector)
        out = self.pooling_layer(x, edge_index, edge_attr, y, batch_vector)

        return out

    def switch_device(self, dev):
        self.dev = dev

def create_model_gnn_class(dev = "cpu"):
    model = GraphNet(num_global_features=8, classification=True, dev = dev).to(dev)
    # model = GraphNet(num_global_features=5, classification=True).to('cpu')
    return model

def create_model_gnn_reg(dev = "cpu"):
    model = GraphNet(num_global_features=8, classification=False, dev = dev).to(dev)
    # model = GraphNet(num_global_features=5, classification=False).to('cpu')
    return model
