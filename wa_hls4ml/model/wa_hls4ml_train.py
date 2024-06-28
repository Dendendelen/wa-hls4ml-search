import torch
import sklearn

import torch_geometric

from torch_geometric import loader as gloader
from torch_geometric import data as gdata

import math
import copy

import model.wa_hls4ml_model as wa_hls4ml_model
import data.wa_hls4ml_data_plot as wa_hls4ml_data_plot

from model.wa_hls4ml_model import save_model

import sys

def train_step(dataloader, model, loss_fn, optimizer, batch_size, is_graph, size):
    ''' Perform the training step on the epoch, going through each batch and performing optimization '''

    # our graph data is in a different format than the numeric data, so we need to have a different iterator for it
    if is_graph:
        iterator = zip(enumerate(dataloader[0]), enumerate(dataloader[1]))
    else:
        iterator = enumerate(dataloader)

    # switch model to train mode
    model.train()
    for item in iterator:
        if is_graph:
            X_en, y_en = item
            batch, X = X_en
            _, y = y_en
        else:
            batch, (X, y) = item

        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred[:,0], y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Report batch results
        if batch % 4 == 0:
            loss_val, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss_val:>7f}  [{current:>5d}/{size:>5d}]")

    # output the last training loss, to plot in history
    return loss.item()

def val_step(dataloader, model, loss_fn, is_graph):
    ''' Perform the validation step, checking performance of the current model on val data to regulate lr '''

    # switch model to evaluation mode
    model.eval()
    num_batches = len(dataloader)
    test_loss = 0

    with torch.no_grad():

        if is_graph:
            iterator = zip(dataloader[0], dataloader[1])
        else:
            iterator = dataloader
        
        for X, y in iterator:
            pred = model(X)
            test_loss += loss_fn(pred[:,0], y).item()

    test_loss /= num_batches
    print(f"Validation Error: Avg loss: {test_loss:>8f} \n")

    return test_loss


def general_train(X_train, y_train, model, loss_function, is_graph, batch_size, test_size, epochs, name, folder, learning_rate, weight_decay, patience, cooldown, factor, min_lr, epsilon):
    ''' Function for performing the training routine given the input data, model, and parameters '''

    # create optimizer and scheduler based on input specifications
    adam = torch.optim.AdamW(lr=learning_rate, params=model.parameters(), weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(adam, patience=patience, cooldown=cooldown, factor=factor, min_lr=min_lr, eps=epsilon)

    # create a train and validation set
    X_only_train, X_val, y_only_train, y_val = sklearn.model_selection.train_test_split(X_train, y_train, random_state=45, test_size=test_size)

    # set up data loading
    if is_graph:
        train_X_dataloader = gloader.DataLoader(X_only_train, batch_size=batch_size)
        train_y_dataloader = gloader.DataLoader(torch.tensor(y_only_train), batch_size=batch_size)
        train_dataloader = (train_X_dataloader, train_y_dataloader)

        val_X_dataloader = gloader.DataLoader(X_val, batch_size=batch_size)
        val_y_dataloader = gloader.DataLoader(torch.tensor(y_val), batch_size=batch_size)
        val_dataloader = (val_X_dataloader, val_y_dataloader)

    else:
        train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_only_train), torch.tensor(y_only_train))
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)

        val_dataset = torch.utils.data.TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)


    directory = folder+'/'+name

    history = {'train':[], 'val':[], 'lr':[]}

    best_loss = math.inf
    best_model = None

    for epoch in range(epochs):
        print("Epoch "+ str(epoch+1) + "/"+str(epochs)+":\n-------------------------------")

        learning_rate = scheduler.get_last_lr()
        print("Learning Rate: "+str(learning_rate))
        history['lr'].append(learning_rate)

        train_loss = train_step(train_dataloader, model, loss_function, adam, batch_size=batch_size, is_graph=is_graph, size=len(X_only_train))
        history['train'].append(train_loss)

        test_loss = val_step(val_dataloader, model, loss_function, is_graph=is_graph)
        history['val'].append(test_loss)
        scheduler.step(test_loss)

        # save the model if it has outcompeted the previous best on the validation set
        if test_loss < best_loss:
            best_loss = test_loss
            best_model = copy.deepcopy(model)

    model.eval()

    wa_hls4ml_data_plot.plot_loss(name, history, folder)
    save_model(best_model, directory)

def train_classifier(X_train, y_train, folder_name, is_graph):
    ''' Train classification model '''
    
    name = 'classification'

    if is_graph:
        model = wa_hls4ml_model.create_model_gnn_class()
    else:
        model = wa_hls4ml_model.create_model_classification()

    loss_function = torch.nn.BCELoss()
    
    test_size = .125
    batch = 512
    epochs = 250

    learning_rate = 0.0001
    weight_decay = 0.001
    patience = 7
    cooldown = 2
    factor = 0.5
    min_lr = 0.00000001
    epsilon = 0.000001

    general_train(X_train, y_train, model, loss_function, is_graph, batch, test_size, epochs, name, folder_name, learning_rate, weight_decay, patience, cooldown, factor, min_lr, epsilon)

def train_regressor(X_train, y_train, output_features, folder_name, is_graph):
    ''' Train regression models for all features '''

    test_size = .125
    batch = 512
    epochs = 250

    learning_rate = 0.01
    weight_decay = 0.001
    patience = 10
    cooldown = 6
    factor = 0.5
    min_lr = 0.000000001
    epsilon = 0.000001

    i = 0
    for feature in output_features:

        # Train regression model to target individual output feature
        print("Training " + feature + "...")
        y_train_feature = y_train[:, i]
        i += 1

        name = 'regression_'+feature

        if not is_graph:
            model = wa_hls4ml_model.create_model_regression_single_feature()
        else:
            model = wa_hls4ml_model.create_model_gnn_reg()
        loss_function = torch.nn.HuberLoss(reduction='mean')

        general_train(X_train, y_train_feature, model, loss_function, is_graph, batch, test_size, epochs, name, folder_name, learning_rate, weight_decay, patience, cooldown, factor, min_lr, epsilon)
