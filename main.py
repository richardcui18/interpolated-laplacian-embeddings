from experiments import run_experiment
import pandas as pd
import numpy as np
from collections import defaultdict
import random
import torch
 
# Set seed here
seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

models = ['gcn', 'mlp', 'gin', 'sage']

s_values = [-1, -0.5, 0, 0.5, 1]
t_values = [-1, -0.5, 0.5, 1]

# # === Graphs with Features ===
# data = defaultdict(dict)
# n_runs = 100
# corruption_levels = [0.0, 0.1, 0.2, 0.5, 0.9]
# # datasets = ["cora", "cornell", "texas", "wisconsin", "wiki"]
# datasets = ['wiki']

# # --------------------------------------------------
# # Step 1. Collect results
# # --------------------------------------------------
# for dataset in datasets:
#     for model in models:
#         for p in corruption_levels:
#             # Baselines
#             for variant in ['none', 'adjacency']:
#                 key = (model, variant)
#                 results = [
#                     run_experiment(dataset, model, [variant], p)[1][variant.capitalize()][-1]
#                     for _ in range(n_runs)
#                 ]
#                 mean, std = np.mean(results), np.std(results)
#                 data[key][p] = (mean, std)

#             # General family
#             for s in s_values:
#                 for t in t_values:
#                     variant = f"$s = {s}, t = {t}$"
#                     key = (model, variant)
#                     results = [
#                         run_experiment(dataset, model, ['general_family'], p, s, t)[1]['General_family'][-1]
#                         for _ in range(n_runs)
#                     ]
#                     mean, std = np.mean(results), np.std(results)
#                     data[key][p] = (mean, std)

# # --------------------------------------------------
# # Step 2. Find best entries for bolding
# # --------------------------------------------------
# def find_best_entries(data, models, corruption_levels, s_values, t_values):
#     best = defaultdict(dict)
#     for model in models:
#         for p in corruption_levels:
#             best_val = -1
#             best_variant = None
#             # baselines
#             for variant in ["none", "adjacency"]:
#                 if (model, variant) in data and p in data[(model, variant)]:
#                     val = data[(model, variant)][p][0]
#                     if val > best_val:
#                         best_val = val
#                         best_variant = (variant, None, None)
#             # grid
#             for s in s_values:
#                 for t in t_values:
#                     variant = f"$s = {s}, t = {t}$"
#                     if (model, variant) in data and p in data[(model, variant)]:
#                         val = data[(model, variant)][p][0]
#                         if val > best_val:
#                             best_val = val
#                             best_variant = (variant, s, t)
#             best[model][p] = best_variant
#     return best

# best_entries = find_best_entries(data, models, corruption_levels, s_values, t_values)

# # --------------------------------------------------
# # Step 3. Make LaTeX table (s rows, t columns, corruption levels as blocks)
# # --------------------------------------------------
# def make_table(data, models, corruption_levels, s_values, t_values, best_entries):
#     latex_lines = []
#     latex_lines.append("\\begin{table}[ht]")
#     latex_lines.append("\\centering")
#     latex_lines.append("\\resizebox{\\textwidth}{!}{%")

#     n_t = len(t_values)
#     n_p = len(corruption_levels)
#     colspec = "ll" + "|".join(["".join(["c"] * n_t) for _ in corruption_levels])
#     latex_lines.append("\\begin{tabular}{%s}" % colspec)

#     # Header row
#     header = ["Model", "Variant"]
#     for p in corruption_levels:
#         perc = int(p * 100)
#         header.append(f"\\multicolumn{{{n_t}}}{{c|}}{{{perc}\\%}}")
#     header[-1] = header[-1].replace("|}}", "}}")  # no trailing bar
#     latex_lines.append(" & ".join(header) + " \\\\")

#     # Sub-header row (t values)
#     subheader = ["", ""]
#     for _ in corruption_levels:
#         subheader.extend([f"$t={t}$" for t in t_values])
#     latex_lines.append(" & ".join(subheader) + " \\\\")
#     latex_lines.append("\\midrule")

#     # Body
#     for model in models:
#         # Baselines
#         for variant in ["none", "adjacency"]:
#             row = [model if variant == "none" else "", variant]
#             for p in corruption_levels:
#                 if (model, variant) in data and p in data[(model, variant)]:
#                     mean, std = data[(model, variant)][p]
#                     cell = f"$%.2f (%.2f)$" % (mean * 100, std * 100)
#                     if best_entries[model][p][0] == variant:
#                         cell = f"$\\mathbf{{{mean * 100:.2f} ({std * 100:.2f})}}$"
#                     row.append("\\multicolumn{%d}{c|}{%s}" % (n_t, cell))
#                 else:
#                     row.append("\\multicolumn{%d}{c|}{--}" % n_t)
#             row[-1] = row[-1].replace("|}", "}")
#             latex_lines.append(" & ".join(row) + " \\\\")

#         # latex_lines.append("\\cmidrule(l){2-%d}" % (2 + n_t * n_p))

#         # s rows
#         for s in s_values:
#             row = ["", f"$s={s}$"]
#             for p in corruption_levels:
#                 for t in t_values:
#                     variant = f"$s = {s}, t = {t}$"
#                     if (model, variant) in data and p in data[(model, variant)]:
#                         mean, std = data[(model, variant)][p]
#                         cell = f"${mean*100:.2f} ({std*100:.2f})$"
#                         if best_entries[model][p][0] == variant:
#                             cell = f"$\\mathbf{{{mean * 100:.2f} ({std * 100:.2f})}}$"
#                         row.append(cell)
#                     else:
#                         row.append("--")
#             latex_lines.append(" & ".join(row) + " \\\\")
#         latex_lines.append("\\midrule")

#     latex_lines.append("\\end{tabular}}")
#     latex_lines.append("\\end{table}")

#     return "\n".join(latex_lines)

# # --------------------------------------------------
# # Step 4. Generate LaTeX code
# # --------------------------------------------------
# latex_code = make_table(data, models, corruption_levels, s_values, t_values, best_entries)
# print(latex_code)

# === SBM and No Features Graph ===
data = defaultdict(dict)
n_runs = 100
# datasets = ['random_core', 'random_affinity'] # SBM
datasets = ["karate", "twitter_congress", 'facebook_ego', 'polblogs'] # no features

# --------------------------------------------------
# Step 1. Collect results
# --------------------------------------------------
for model in models:
    for dataset in datasets:
        # Base variants
        for variant in ['none', 'adjacency']:
            key = (model, variant)
            results = [
                run_experiment(dataset, model, [variant], 0.0)[1][variant.capitalize()][-1]
                for _ in range(n_runs)
            ]
            mean, std = np.mean(results), np.std(results)
            data[key][dataset] = (mean, std)

        # General family (s,t) grid
        for s in s_values:
            for t in t_values:
                variant = f"$s = {s}, t = {t}$"
                key = (model, variant)
                results = [
                    run_experiment(dataset, model, ['general_family'], 0.0, s, t)[1]['General_family'][-1]
                    for _ in range(n_runs)
                ]
                mean, std = np.mean(results), np.std(results)
                data[key][dataset] = (mean, std)

# --------------------------------------------------
# Step 2. Helper to find best entry
# --------------------------------------------------
def find_best_entries(data, models, datasets, s_values, t_values):
    best = defaultdict(dict)
    for model in models:
        for dataset in datasets:
            best_val = -1
            best_variant = None
            # check baselines
            for variant in ["none", "adjacency"]:
                if (model, variant) in data and dataset in data[(model, variant)]:
                    val = data[(model, variant)][dataset][0]
                    if val > best_val:
                        best_val = val
                        best_variant = (variant, None, None)
            # check grid
            for s in s_values:
                for t in t_values:
                    variant = f"$s = {s}, t = {t}$"
                    if (model, variant) in data and dataset in data[(model, variant)]:
                        val = data[(model, variant)][dataset][0]
                        if val > best_val:
                            best_val = val
                            best_variant = (variant, s, t)
            best[model][dataset] = best_variant
    return best

best_entries = find_best_entries(data, models, datasets, s_values, t_values)

# --------------------------------------------------
# Step 3. Make LaTeX table (s rows, t columns)
# --------------------------------------------------
def make_table(data, models, datasets, s_values, t_values, best_entries):
    latex_lines = []
    latex_lines.append("\\begin{table}[ht]")
    latex_lines.append("\\centering")
    latex_lines.append("\\resizebox{\\textwidth}{!}{%")

    # Column definition: 2 left columns + dataset blocks with vertical separators
    n_t = len(t_values)
    n_d = len(datasets)
    colspec = "ll" + "|".join(["".join(["c"] * n_t) for _ in datasets])
    latex_lines.append("\\begin{tabular}{%s}" % colspec)

    # Header row
    header = ["Model", "Variant"]
    for ds in datasets:
        header.append(f"\\multicolumn{{{n_t}}}{{c|}}{{{ds}}}")
    header[-1] = header[-1].replace("|}}", "}}")  # no trailing vertical bar
    latex_lines.append(" & ".join(header) + " \\\\")

    # Sub-header row (t values)
    subheader = ["", ""]
    for _ in datasets:
        subheader.extend([f"$t={t}$" for t in t_values])
    latex_lines.append(" & ".join(subheader) + " \\\\")
    latex_lines.append("\\midrule")

    # Body
    for model in models:
        # Baselines
        for variant in ["none", "adjacency"]:
            row = [model if variant == "none" else "", variant]
            for ds in datasets:
                if (model, variant) in data and ds in data[(model, variant)]:
                    mean, std = data[(model, variant)][ds]
                    cell = f"$%.2f (%.2f)$" % (mean * 100, std * 100)
                    if best_entries[model][ds][0] == variant:
                        cell = f"$\\mathbf{{{mean * 100:.2f} ({std * 100:.2f})}}$"
                    row.append("\\multicolumn{%d}{c|}{%s}" % (n_t, cell))
                else:
                    row.append("\\multicolumn{%d}{c|}{--}" % n_t)
            row[-1] = row[-1].replace("|}", "}")  # remove trailing bar
            latex_lines.append(" & ".join(row) + " \\\\")

        latex_lines.append("\\cmidrule(l){2-%d}" % (2 + n_t * n_d))

        # s rows
        for s in s_values:
            row = ["", f"$s={s}$"]
            for ds in datasets:
                for t in t_values:
                    variant = f"$s = {s}, t = {t}$"
                    if (model, variant) in data and ds in data[(model, variant)]:
                        mean, std = data[(model, variant)][ds]
                        cell = f"${mean*100:.2f} ({std*100:.2f})$"
                        if best_entries[model][ds][0] == variant:
                            cell = f"$\\mathbf{{{mean * 100:.2f} ({std * 100:.2f})}}$"
                        row.append(cell)
                    else:
                        row.append("--")
            latex_lines.append(" & ".join(row) + " \\\\")
        latex_lines.append("\\midrule")

    # End table
    latex_lines.append("\\end{tabular}}")
    latex_lines.append("\\end{table}")

    return "\n".join(latex_lines)

# --------------------------------------------------
# Step 4. Generate LaTeX code
# --------------------------------------------------
latex_code = make_table(data, models, datasets, s_values, t_values, best_entries)
print(latex_code)