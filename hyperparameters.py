"""
Define the hyperparameters used via a dictionary and a getter
"""

def get_hyperparameters(model_name, dataset, features, num_classes, k):
    hyperparameters = {
        "batch_size": 100,
        "learning_rate": 0.01,
        "hidden_size": 16,
        "input_size": features.size(1),
        "num_classes": num_classes,
        "k": k
    }

    if model_name == "gcn":
        hyperparameters['num_epochs'] = 150
    elif model_name == "mlp":
        hyperparameters['num_epochs'] = 150
    # elif model_name == "gat":
    #     hyperparameters['num_epochs'] = 150
    elif model_name == "gin":
        hyperparameters['num_epochs'] = 150
    elif model_name == "sage":
        hyperparameters['num_epochs'] = 150

    return hyperparameters