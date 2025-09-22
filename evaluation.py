import torch
## Here we evaluate the model

@torch.no_grad()
def evaluate_model(trained_model, features, data, test_mask):
    trained_model.eval()
    edge_input = data.edge_index
    _, z = trained_model(features, edge_input)
    pred = z.argmax(dim=1)
    acc = accuracy(pred[test_mask], data.y[test_mask])
    return acc.item()

# Accuracy Function
def accuracy(pred_y, y):
    return (pred_y == y).sum().float() / len(y)