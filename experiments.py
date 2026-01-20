"""
An experiment, in our context, only depend on two input instructions:

1. Dataset used (e.g. SBM, LastFM Asia)
2. Model used (e.g. MLP, GCN)
3. Features used (e.g. None, Adjacency, Laplacian)
"""

from datasets import get_dataloaders
from hyperparameters import get_hyperparameters
from models import get_model
from train import train_model
from evaluation import evaluate_model
from logger import logToFile
from features import get_features
from graphing import draw_comparison_graph
import os

# An experiment is run in 6 simple steps
# We load the appropriate dataset
# We load the appropriate hyperparameters
# We load the appropriate model
# We train the model
# We evaluate the model
# We log the results
def run_experiment(dataset, model_name, feature_names, p, t = None, num_layers = None):
    k = 10
    net_type = 'sign_net'
    print(f"Running p = {p}...")
    
    data, train_mask, test_mask, num_classes = get_dataloaders(dataset)

    if net_type == "basis_net":
        n = data.num_nodes
        k = n

    accuracies_dict_train = {}
    accuracies_dict_test = {}

    for feature_name in feature_names:
        if feature_name == "none":
            k = 0
        if t is not None:
            print(f"Running with feature {feature_name}, t={t}...")
        else:
            print(f"Running with feature {feature_name}...")
        features = get_features(data, feature_name, k, p, t)
        hyperparams = get_hyperparameters(model_name, data, features, num_classes, k)

        if model_name == "mlp":
            model = get_model("mlp", hyperparams, k, net_type)
        elif model_name == "gcn":
            model = get_model("gcn", hyperparams, k, net_type, num_layers=num_layers)
        elif model_name == "gin":
            model = get_model("gin", hyperparams, k, net_type)
        # elif model_name == "gat":
        #     model = get_model("gat", hyperparams, k, net_type)
        elif model_name == "sage":
            model = get_model("sage", hyperparams, k, net_type)
        else:
            raise Exception("model {} not supported".format(model_name))

        trained_model, train_acc, accuracy_list_train, accuracy_list_test = train_model(model, hyperparams, features, data, train_mask, test_mask)
        test_acc = evaluate_model(trained_model, features, data, test_mask)
        filename = "logs/{}_{}_{}.txt".format(dataset, model_name, feature_name)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        if t is not None:
            logToFile(filename, """p: {}, t: {}, training accuracy: {}, testing accuracy: {}\n""".format(p, t, round(train_acc*100, 2), round(test_acc*100, 2)))
        else:
            logToFile(filename, """p: {}, training accuracy: {}, testing accuracy: {}\n""".format(p, round(train_acc*100, 2), round(test_acc*100, 2)))

        accuracies_dict_train[feature_name.capitalize()] = accuracy_list_train
        accuracies_dict_test[feature_name.capitalize()] = accuracy_list_test

    draw_comparison_graph(accuracies_dict_train, model_name, dataset, "train")
    draw_comparison_graph(accuracies_dict_test, model_name, dataset, "test")

    return accuracies_dict_train, accuracies_dict_test
