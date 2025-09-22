import torch
import logging
from evaluation import accuracy
from evaluation import evaluate_model
import random
import numpy as np
import torch

# here we train the model
def train_model(model, hyperparams, features, data, train_mask, test_mask):

    # unpack hyperparameters
    num_epochs = hyperparams["num_epochs"]
    learning_rate = hyperparams["learning_rate"]

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)

    criterion = torch.nn.CrossEntropyLoss()

    accuracy_list_train = []
    accuracy_list_test = []

    # perform training for each epoch, and print out performance accordingly
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, features, data, train_mask, optimizer, criterion)
        test_acc = evaluate_model(model, features, data, test_mask)
        if epoch % 10 == 0:
            print(f"Epoch {epoch:>3} | "
                f"Train Accs: {train_acc*100:.2f}% | "
                f"Test Accs: {test_acc*100:.2f}%")
        
        accuracy_list_train.append(train_acc)
        accuracy_list_test.append(test_acc)

    return model, train_acc, accuracy_list_train, accuracy_list_test


# Training Function
def train_epoch(model, features, data, train_mask, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    edge_input = data.edge_index
    h, z = model(features, edge_input)
    loss = criterion(z[train_mask], data.y[train_mask])
    acc = accuracy(z[train_mask].argmax(dim=1), data.y[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item(), acc.item()
