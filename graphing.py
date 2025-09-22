import matplotlib.pyplot as plt
import numpy as np
import os

def draw_comparison_graph(accuracies, model, dataset, train_test):
    # Plot Accuracy Results
    plt.figure(figsize=(10, 6))
    for label, accs in accuracies.items():
        plt.plot(accs, label=label)
    plt.xlabel("Epoch")
    plt.ylabel(f"{train_test.capitalize()} Accuracy")
    plt.title(f"{model.upper()} Comparison between Features on {dataset.capitalize()} during {train_test.capitalize()}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(os.path.dirname(f'graphs/{dataset}/{model}_embedding_comparison_{train_test}.png'), exist_ok=True)
    plt.savefig(f"graphs/{dataset}/{model}_embedding_comparison_{train_test}.png", dpi=300)
    print(f"Plot saved as 'graphs/{dataset}/{model}_embedding_comparison_{train_test}.png'")
    plt.close()


def draw_median_range_graph(results_list, model_name, dataset, split="train"):
    feature_names = results_list[0].keys()
    num_epochs = len(next(iter(results_list[0].values())))

    plt.figure(figsize=(10, 6))

    for feature in feature_names:
        all_runs = np.array([run[feature] for run in results_list])
        median = np.median(all_runs, axis=0)
        min_vals = np.min(all_runs, axis=0)
        max_vals = np.max(all_runs, axis=0)
        
        epochs = np.arange(len(median))
        plt.plot(epochs, median, label=feature)
        plt.fill_between(epochs, min_vals, max_vals, alpha=0.3)

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'{split.capitalize()} Accuracy (Median ± Range)\n{model_name.upper()} on {dataset}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(os.path.dirname(f'graphs/{dataset}/{model_name}_{split}_median_range.png'), exist_ok=True)
    plt.savefig(f"graphs/{dataset}/{model_name}_{split}_median_range.png", dpi=300)
    plt.show()
