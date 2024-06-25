import torch
import sklearn

import math

import wa_hls4ml_model
import wa_hls4ml_data_plot

from wa_hls4ml_model import save_model


def train_step(dataloader, model, loss_fn, optimizer, batch_size):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss

        pred = model(X)
        loss = loss_fn(pred[:,0], y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 2 == 0:
            loss_val, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss_val:>7f}  [{current:>5d}/{size:>5d}]")

    return loss.item()

def val_step(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred[:,0], y).item()

    test_loss /= num_batches
    print(f"Validation Error: Avg loss: {test_loss:>8f} \n")

    return test_loss

def train_classifier(X_train, y_train, folder_name):
    ''' Train classification model '''

    #TODO: make this an argument
    model = wa_hls4ml_model.create_model_classification()
    # model = wa_hls4ml_model.create_model_gnn()

    batch = 256

    loss_function = torch.nn.BCELoss()
    adam = torch.optim.AdamW(lr=0.001, params=model.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(adam, patience=7, cooldown=2, factor=0.5, min_lr=0.00000001, eps=0.000001)

    # create a train and validation set
    X_only_train, X_val, y_only_train, y_val = sklearn.model_selection.train_test_split(X_train, y_train, random_state=45, test_size=0.125)

    train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_only_train), torch.tensor(y_only_train))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch)

    val_dataset = torch.utils.data.TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch)

    directory = folder_name+'/classification' 

    history = {'train':[], 'val':[]}

    epochs = 400
    # prev_best_loss = math.inf
    for epoch in range(epochs):
        print("Epoch "+ str(epoch+1) + "/"+str(epochs)+":\n-------------------------------")
        print("Learning Rate: "+str(scheduler.get_last_lr()))
        train_loss = train_step(train_dataloader, model, loss_function, adam, batch_size=batch)
        history['train'].append(train_loss)

        test_loss = val_step(val_dataloader, model, loss_function)
        history['val'].append(test_loss)
        scheduler.step(test_loss)

    model.eval()

    wa_hls4ml_data_plot.plot_loss('classification', history, folder_name)
    save_model(model, directory)

def train_regressor(X_train, y_train, output_features, graph, folder_name):
    ''' Train regression models for all features '''

    i = 0
    for feature in output_features:

        # Train regression model to target individual output feature
        print("Training " + feature + "...")
        y_train_feature = y_train[:, i]
        i += 1

        # if not graph:
        model = wa_hls4ml_model.create_model_regression_single_feature()
        # else:
        #     model = wa_hls4ml_model.create_model_gnn()

        batch = 128

        adam = torch.optim.AdamW(lr=0.005, weight_decay=0.004, params=model.parameters())
        loss_function = torch.nn.HuberLoss(reduction='mean')
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(adam, patience=8, factor=0.5, min_lr=0.00000001, cooldown=3, eps=0.0001)

        # create a train and validation set
        X_only_train, X_val, y_only_train, y_val = sklearn.model_selection.train_test_split(X_train, y_train_feature, random_state=45, test_size=0.125)


        train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_only_train), torch.tensor(y_only_train))
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch)

        val_dataset = torch.utils.data.TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch)

        history = {'train':[], 'val':[]}

        epochs = 250
        for epoch in range(epochs):
            print("Epoch "+ str(epoch+1) + ":\n-------------------------------")
            print("Learning Rate: "+str(scheduler.get_last_lr()))

            train_loss = train_step(train_dataloader, model, loss_function, adam, batch)
            history['train'].append(train_loss)

            test_loss = val_step(val_dataloader, model, loss_function)
            history['val'].append(test_loss)
            scheduler.step(test_loss)

        model.eval()

        directory = folder_name+'/regression_'+ feature
        save_model(model, directory)

        wa_hls4ml_data_plot.plot_loss('regression_'+feature, history, folder_name)
