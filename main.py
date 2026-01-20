from experiments import run_experiment
import pandas as pd
import numpy as np
from collections import defaultdict
import random
import torch
from evaluation import spectral_alignment_scores
from datasets import get_dataloaders
 
# Set seed here
seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# models = ['gcn', 'mlp', 'gin', 'sage']
models = ['gcn', 'mlp']
num_layers = 5

print("Num layers in GCN", num_layers)

t_values = [round(x, 2) for x in np.arange(0.0, 1.01, 0.1)]

# # === Graphs with Features ===
# data = defaultdict(dict)
# n_runs = 5
# corruption_levels = [0.0, 0.1, 0.2, 0.5, 0.9]
# datasets = ["cora", "cornell", "texas", "wisconsin", "wiki"]

# # ----------------------------------------------------------------------
# # Step 0: Find optimal t by dataset
# # ----------------------------------------------------------------------
# optimal_t_by_dataset = {}

# for ds_name in datasets:
#     data_obj,train_mask,_,_ = get_dataloaders(ds_name)
    
#     # Compute spectral scores across grid
#     corr_list = []
#     spear_list = []
#     for t in t_values:
#         out = spectral_alignment_scores(data_obj, t, k=10, observed_mask=train_mask)
#         corr_list.append(out["correlation_score"])
#         spear_list.append(out["spearman_score"])
        
#     corr_arr = np.array(corr_list)
#     spear_arr = np.array(spear_list)

#     if np.all(np.isneginf(corr_arr)) and np.all(np.isneginf(spear_arr)):
#         optimal_t_by_dataset[ds_name] = (None, None, False)
#         print(f"[WARN] spectral selection produced no valid scores for '{ds_name}'. Falling back later.")
#     else:
#         t_corr = float(t_values[np.nanargmax(corr_arr)])
#         t_spear = float(t_values[np.nanargmax(spear_arr)])
#         optimal_t_by_dataset[ds_name] = (t_corr, t_spear, True)
#         print(f"[INFO] dataset={ds_name} spectral t_corr={t_corr}, t_spear={t_spear}")


# # --------------------------------------------------
# # Step 1. Collect results
# # --------------------------------------------------
# for dataset in datasets:
#     optimal_ts = optimal_t_by_dataset[dataset]
#     for model in models:
#         for p in corruption_levels:
#             # Baselines
#             for variant in ['none', 'adjacency', 'laplacian']:
#                 key = (model, variant)
#                 results = [
#                     run_experiment(dataset, model, [variant], p, num_layers=num_layers)[1][variant.capitalize()][-1]
#                     for _ in range(n_runs)
#                 ]
#                 mean, std = np.mean(results), np.std(results)
#                 data[key][p] = (mean, std)

#             # General family
#             for t_metric in ["Correlation", "Spearman"]:
#                 if t_metric == "Correlation":
#                     t = optimal_ts[0]
#                 elif t_metric == "Spearman":
#                     t = optimal_ts[1]
#                 variant = f"Optimal $t$ ({t_metric})"
#                 key = (model, variant)
#                 results = [
#                     run_experiment(dataset, model, ['general_family'], p, t=t, num_layers=num_layers)[1]['General_family'][-1]
#                     for _ in range(n_runs)
#                 ]
#                 mean, std = np.mean(results), np.std(results)
#                 data[key][p] = (mean, std)

# # --------------------------------------------------
# # Step 2. Find best entries for bolding
# # --------------------------------------------------
# def find_best_entries(data, models, corruption_levels):
#     best = defaultdict(dict)
#     for model in models:
#         for p in corruption_levels:
#             best_val = -1
#             best_variant = None
#             # baselines
#             for variant in ["none", "adjacency", "laplacian"]:
#                 if (model, variant) in data and p in data[(model, variant)]:
#                     val = data[(model, variant)][p][0]
#                     if val > best_val:
#                         best_val = val
#                         best_variant = (variant)
#             # General family
#             for variant in ["Optimal $t$ (Correlation)", "Optimal $t$ (Spearman)"]:
#                 if (model, variant) in data and p in data[(model, variant)]:
#                     val = data[(model, variant)][p][0]
#                     if val > best_val:
#                         best_val = val
#                         best_variant = (variant)
#             best[model][p] = best_variant
#     return best

# best_entries = find_best_entries(data, models, corruption_levels)

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
n_runs = 5
datasets = ['random_core', 'random_affinity'] # SBM
# datasets = ["karate", "twitter_congress", 'facebook_ego', 'polblogs'] # no features

# ----------------------------------------------------------------------
# Step 0: Find optimal t by dataset
# ----------------------------------------------------------------------
optimal_t_by_dataset = {}

for ds_name in datasets:
    data_obj,train_mask,_,_ = get_dataloaders(ds_name)
    
    # Compute spectral scores across grid
    corr_list = []
    spear_list = []
    for t in t_values:
        out = spectral_alignment_scores(data_obj, t, k=10, observed_mask=train_mask)
        corr_list.append(out["correlation_score"])
        spear_list.append(out["spearman_score"])
        
    corr_arr = np.array(corr_list)
    spear_arr = np.array(spear_list)

    if np.all(np.isneginf(corr_arr)) and np.all(np.isneginf(spear_arr)):
        optimal_t_by_dataset[ds_name] = (None, None, False)
        print(f"[WARN] spectral selection produced no valid scores for '{ds_name}'. Falling back later.")
    else:
        t_corr = float(t_values[np.nanargmax(corr_arr)])
        t_spear = float(t_values[np.nanargmax(spear_arr)])
        optimal_t_by_dataset[ds_name] = (t_corr, t_spear, True)
        print(f"[INFO] dataset={ds_name} spectral t_corr={t_corr}, t_spear={t_spear}")


# --------------------------------------------------
# Step 1. Collect results
# --------------------------------------------------
for model in models:
    for dataset in datasets:
        optimal_ts = optimal_t_by_dataset[dataset]

        # Base variants
        for variant in ['none', 'adjacency', 'laplacian']:
            key = (model, variant)
            results = [
                run_experiment(dataset, model, [variant], 0.0, num_layers=num_layers)[1][variant.capitalize()][-1]
                for _ in range(n_runs)
            ]
            mean, std = np.mean(results), np.std(results)
            data[key][dataset] = (mean, std)

        # General family
        for t_metric in ["Correlation", "Spearman"]:
            if t_metric == "Correlation":
                t = optimal_ts[0]
            elif t_metric == "Spearman":
                t = optimal_ts[1]
            variant = f"Optimal $t$ ({t_metric})"
            key = (model, variant)
            results = [
                run_experiment(dataset, model, ['general_family'], 0.0, t=t, num_layers=num_layers)[1]['General_family'][-1]
                for _ in range(n_runs)
            ]
            mean, std = np.mean(results), np.std(results)
            data[key][dataset] = (mean, std)


# --------------------------------------------------
# Step 2. Helper to find best entry
# --------------------------------------------------
def find_best_entries(data, models, datasets):
    best = defaultdict(dict)
    for model in models:
        for dataset in datasets:
            best_val = -1
            best_variant = None
            # check baselines
            for variant in ["none", "adjacency", "laplacian"]:
                if (model, variant) in data and dataset in data[(model, variant)]:
                    val = data[(model, variant)][dataset][0]
                    if val > best_val:
                        best_val = val
                        best_variant = variant
            # General family
            for variant in ["Optimal $t$ (Correlation)", "Optimal $t$ (Spearman)"]:
                if (model, variant) in data and dataset in data[(model, variant)]:
                    val = data[(model, variant)][dataset][0]
                    if val > best_val:
                        best_val = val
                        best_variant = variant
            best[model][dataset] = best_variant
    return best


best_entries = find_best_entries(data, models, datasets)

# --------------------------------------------------
# Step 3. Make LaTeX table
# --------------------------------------------------
def make_table(data, models, datasets, best_entries):
    latex_lines = []
    latex_lines.append("\\begin{table}[ht]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption {Testing accuracy (\\%) of different models augmented with ILEs.}")
    latex_lines.append("\\resizebox{\\textwidth}{!}{%")


    # Column definition
    n_d = len(datasets)
    colspec = "ll|" + "c"*n_d
    latex_lines.append("\\begin{tabular}{%s}" % colspec)
    latex_lines.append("\\toprule")

    # Header row
    header = ["\\textbf{Model}", "\\textbf{Variant}"]
    for ds in datasets:
        header.append(f"\\textbf{{{ds}}}")
    latex_lines.append(" & ".join(header) + " \\\\")
    latex_lines.append("\\midrule")

    # Body
    # for model in models:
    #     # Baselines
    #     for variant in ["none", "adjacency", "laplacian"]:
    #         row = [model if variant == "none" else "", variant]
    #         for ds in datasets:
    #             if (model, variant) in data and ds in data[(model, variant)]:
    #                 mean, std = data[(model, variant)][ds]
    #                 cell = f"$%.2f (%.2f)$" % (mean * 100, std * 100)
    #                 if best_entries[model][ds][0] == variant:
    #                     cell = f"$\\mathbf{{{mean * 100:.2f} ({std * 100:.2f})}}$"
    #                 row.append("\\multicolumn{%d}{c|}{%s}" % (n_t, cell))
    #             else:
    #                 row.append("\\multicolumn{%d}{c|}{--}" % n_t)
    #         row[-1] = row[-1].replace("|}", "}")  # remove trailing bar
    #         latex_lines.append(" & ".join(row) + " \\\\")

    #     latex_lines.append("\\cmidrule(l){2-%d}" % (2 + n_t * n_d))

    #     # s rows
    #     for s in s_values:
    #         row = ["", f"$s={s}$"]
    #         for ds in datasets:
    #             for t in t_values:
    #                 variant = f"$s = {s}, t = {t}$"
    #                 if (model, variant) in data and ds in data[(model, variant)]:
    #                     mean, std = data[(model, variant)][ds]
    #                     cell = f"${mean*100:.2f} ({std*100:.2f})$"
    #                     if best_entries[model][ds][0] == variant:
    #                         cell = f"$\\mathbf{{{mean * 100:.2f} ({std * 100:.2f})}}$"
    #                     row.append(cell)
    #                 else:
    #                     row.append("--")
    #         latex_lines.append(" & ".join(row) + " \\\\")
    #     latex_lines.append("\\midrule")

    for model in models:
        first_variant = True
        for variant in ["None", "Adjacency", "Laplacian", "Optimal $t$ (Correlation)", "Optimal $t$ (Spearman)"]:
            model_cell = model if first_variant else ""
            # gather per-dataset cell strings
            cells = [model.upper(), variant] if first_variant else ["", variant]
            for ds in datasets:
                key = (model, variant.lower() if variant in ["None", "Adjacency", "Laplacian"] else variant)
                # For baseline variants we stored keys as lowercase names ('none','adjacency','laplacian')
                if variant in ["None", "Adjacency", "Laplacian"]:
                    lookup_key = (model, variant.lower())
                else:
                    lookup_key = (model, variant)

                if lookup_key in data and ds in data[lookup_key]:
                    mean, std = data[lookup_key][ds]
                    if mean is None:
                        cells.append("--")
                    else:
                        val_str = f"$%.2f (%.2f)$" % (mean * 100, std * 100)
                        # bold if empirical best
                        # if best_entries.get(model, {}).get(ds, (None,))[0] == lookup_key[1] if best_entries.get(model, {}).get(ds) else False:
                        #     cells.append(f"\\textbf{{{val_str}}}")
                        # else:
                        #     cells.append(val_str)

                        best_variant = best_entries.get(model, {}).get(ds)
                        if best_variant == lookup_key[1]:
                            val_str = f"$\\mathbf{{{mean * 100:.2f} ({std * 100:.2f})}}$"
                            cells.append(val_str)
                        else:
                            cells.append(val_str)

                else:
                    cells.append("--")

            latex_lines.append(" & ".join(cells) + " \\\\")
            # latex_lines.append(f"{model_cell} & {variant} & {cells[0]} & {cells[1]} \\\\")
            first_variant = False
        latex_lines.append("\\midrule")


    # End table
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}}")
    latex_lines.append("\\end{table}")
    table_tex = "\n".join(latex_lines)

    return table_tex

# --------------------------------------------------
# Step 4. Generate LaTeX code
# --------------------------------------------------
table_tex = make_table(data, models, datasets, best_entries)
print(table_tex)
print()
print(optimal_t_by_dataset)
