import ast
from argparse import ArgumentParser
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

tqdm.pandas()
sns.set_theme()
sns.set_style("whitegrid")


def save_plot(path, title=None, xlabel=None, ylabel=None, legend=True):
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if legend:
        plt.legend()
    if title is not None:
        plt.title(title)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def plot_matrix(matrix, labels, x_label, y_label, vmax=None):
    with plt.rc_context(rc={"font.size": 60}):
        fig, ax = plt.subplots()
        fig.set_size_inches(20, 20)
        if vmax is not None:
            im = ax.imshow(matrix, origin='lower', vmin=0, vmax=vmax)
        else:
            im = ax.imshow(matrix, origin='lower')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.5)
        fig.colorbar(im, cax=cax, orientation='vertical')
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ticks = np.arange(0, len(labels), 8)
        ticks_labels = [labels[i] for i in ticks]
        ax.set_xticks(ticks, labels=ticks_labels)
        ax.set_yticks(ticks, labels=ticks_labels)


def plot_layer_matrix(generations, target_classification, path):
    success_matrix = (generations.groupby(["source_layer", "target_layer"])["classification"].
                      value_counts(normalize=True).
                      unstack("classification").
                      fillna(0)
                      * 100)
    matrix = success_matrix[target_classification].unstack("target_layer").fillna(0)
    plot_matrix(matrix.to_numpy(), matrix.columns, "Target Layer", "Source Layer")
    save_plot(path, legend=False)


def get_entity_counts(df):
    return pd.DataFrame({
        "e2": {
            "e1": df["e1_e2_entity_layer"].count() / len(df),
            "last": df["last_e2_entity_layer"].count() / len(df),
        },
        "e3": {
            "last": df["last_e3_entity_layer"].count() / len(df),
        }
    })


def get_sublayer_counts(df, target):
    return pd.DataFrame({
        "proportion": {
            "attention": df[f"attention_{target}_projection_layers"].count() / len(df),
            "mlp": df[f"mlp_{target}_projection_layers"].count() / len(df)
        },
        "mean layer": {
            "attention": df[f"attention_{target}_projection_layers"].explode().mean(),
            "mlp": df[f"mlp_{target}_projection_layers"].explode().mean()
        }
    })


def get_stage_layers(df):
    return pd.DataFrame({
        "stage": ["e1_e2_entity_layer", "info_prop", "last_e3_entity_layer", "mlp_prediction_projection_layers"],
        "proportion": [df["e1_e2_entity_layer"].count() / len(df),
                       df["info_prop"].count() / len(df),
                       df["last_e3_entity_layer"].count() / len(df),
                       df["mlp_prediction_projection_layers"].count() / len(df)],
        "layer": [df["e1_e2_entity_layer"].mean(), df["info_prop"].mean(), df["last_e3_entity_layer"].mean(),
                  df["mlp_prediction_projection_layers"].mean()],
    })


def plot_stage_boxplots(correct_df, incorrect_df, path):
    correct_df = correct_df.copy()
    incorrect_df = incorrect_df.copy()

    for df in [correct_df, incorrect_df]:
        for col in ["e1_e2_entity_layer", "last_e1_attention_knockout_layers", "attention_e2_projection_layers",
                    "last_e2_entity_layer", "last_e3_entity_layer", "mlp_prediction_projection_layers"]:
            df[col] = df[col].apply(lambda x: min(x) if type(x) is list and len(x) > 0 else np.nan)

    correct_df["info_prop"] = correct_df.apply(lambda row: np.nanmean(
        [row["last_e1_attention_knockout_layers"], row["attention_e2_projection_layers"], row["last_e2_entity_layer"]]),
                                               axis=1)
    correct_data = pd.melt(correct_df, value_vars=["e1_e2_entity_layer", "info_prop", "last_e3_entity_layer",
                                                   "mlp_prediction_projection_layers"], value_name='value',
                           var_name='stage')
    correct_data["Two-Hop Query Answer"] = "Correct"
    print("Correct setting stages:")
    print(get_stage_layers(correct_df))

    incorrect_df["info_prop"] = incorrect_df.apply(lambda row: np.nanmean(
        [row["last_e1_attention_knockout_layers"], row["attention_e2_projection_layers"], row["last_e2_entity_layer"]]),
                                                   axis=1)
    incorrect_data = pd.melt(incorrect_df, value_vars=["e1_e2_entity_layer", "info_prop", "last_e3_entity_layer",
                                                       "mlp_prediction_projection_layers"], value_name='value',
                             var_name='stage')
    incorrect_data["Two-Hop Query Answer"] = "Incorrect"
    print("Incorrect setting stages:")
    print(get_stage_layers(incorrect_df))

    combined = pd.concat([correct_data, incorrect_data]).reset_index()
    labels = {
        "e1_e2_entity_layer": "1st Hop Resolved",
        "info_prop": "Info Propagation",
        "last_e3_entity_layer": "2nd Hop Resolved",
        "mlp_prediction_projection_layers": "Prediction Extracted"
    }
    combined = combined.replace(labels)
    with sns.plotting_context("paper", font_scale=1.5):
        fig, ax = plt.subplots()
        hatches = ['//', '..']
        sns.boxplot(x="stage", y="value", data=combined, hue="Two-Hop Query Answer", gap=0.1, ax=ax)
        ax.set_xlabel(None)
        ax.set_xticklabels(labels=labels.values(), rotation=45, ha="right", rotation_mode='anchor')
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=2, title="Two-Hop Query Answer")

        patches = [patch for patch in ax.patches if type(patch) is matplotlib.patches.PathPatch]
        h = np.repeat(hatches, (len(patches) // len(hatches)))
        for patch, hatch in zip(patches, h):
            patch.set_hatch(hatch)
            fc = patch.get_facecolor()
            patch.set_edgecolor(fc)
            patch.set_facecolor('none')

        for lp, hatch in zip(ax.legend().get_patches(), hatches):
            lp.set_hatch(hatch)
            fc = lp.get_facecolor()
            lp.set_edgecolor(fc)
            lp.set_facecolor('none')

        sns.move_legend(ax, loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=2, title="Two-Hop Query Answer")

        save_plot(f"{path}/stages_boxplot.pdf", ylabel="Layer", legend=False)


def plot_series(series, label, layers, dropna=True, ax=None, linestyle="-"):
    data = series.value_counts(normalize=True, dropna=dropna).sort_index().reindex(range(layers), fill_value=0) * 100
    if ax is not None:
        return sns.lineplot(data, ax=ax, label=label, linestyle=linestyle)
    return sns.lineplot(data=data, label=label, linestyle=linestyle)


def plot_min_entity_description_layers(df, path, layers):
    with sns.plotting_context("paper", font_scale=1.5):
        fig, ax = plt.subplots()
        plot_series(df["e1_e2_entity_layer"].apply(lambda x: min(x) if type(x) is list else np.nan),
                    "$e_2$ decoded from $t_1$",
                    layers,
                    dropna=False,
                    ax=ax)
        plot_series(df["last_e3_entity_layer"].apply(lambda x: min(x) if type(x) is list else np.nan),
                    "$e_3$ decoded from $t_2$",
                    layers,
                    dropna=False,
                    ax=ax,
                    linestyle="--")
        sns.move_legend(ax, loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2)
        save_plot(path, xlabel="Layer", ylabel="Percentage of Cases", legend=False)



def calc_backpatching_oracle_success(df, source):
    return df[f"{source}_original_activation_patching_layer"].count() / len(df)


def calc_backpatching_success(layers):
    return pd.DataFrame({
        "Correct": {
            "e1": calc_backpatching_oracle_success(layers["correct"], "e1"),
            "last": calc_backpatching_oracle_success(layers["correct"], "last"),
        },
        "Incorrect": {
            "e1": calc_backpatching_oracle_success(layers["incorrect"], "e1"),
            "last": calc_backpatching_oracle_success(layers["incorrect"], "last"),
        }
    })


def load_layers(path):
    df = pd.read_csv(path, index_col=0)
    for col in df.columns:
        if col not in ["id", "composition_correct"]:
            df[col] = df[col].apply(lambda x: ast.literal_eval(x) if type(x) is not float else x)
    return df


def load_backpatching(path):
    df = pd.read_csv(path, index_col=0)
    df.loc[df["source_layer"] <= df["target_layer"], "classification"] = "masked"
    return df.copy()


def load_entity_description(path, source, seeds=3):
    gens = []
    for seed in range(seeds):
        csv_path = f"{path}/{source}_entity_description_s{seed}_classified.csv"
        gens.append(pd.read_csv(csv_path, index_col=0))
    generations = pd.concat(gens).drop_duplicates(
        subset=["id", "source_layer", "target_layer", "classification"]).sort_values(
        ["id", "source_layer", "target_layer"])
    return generations


def main(args):
    layer_count = {
        "meta-llama/Llama-2-7b-hf": 32,
        "meta-llama/Llama-2-13b-hf": 40,
        "meta-llama/Meta-Llama-3-8B": 32,
        "meta-llama/Meta-Llama-3-70B": 80
    }
    layers = {}
    for model in args.models:
        print(f"Processing {model} stages")
        dataset_path = f"{args.datasets_dir}/{model}"
        plots_path = f"{args.output_path}/{model}"
        layers[model] = {
            "correct": load_layers(f"{dataset_path}/composition_correct/layers.csv"),
            "incorrect": load_layers(f"{dataset_path}/composition_incorrect/layers.csv")
        }
        print(f"Correct setting example count: {len(layers[model]['correct'])}")
        print(f"Incorrect setting example count: {len(layers[model]['incorrect'])}")
        plot_stage_boxplots(layers[model]["correct"], layers[model]["incorrect"], plots_path)
        print(f"Correct setting prediction in sublayers:\n"
              f"{get_sublayer_counts(layers[model]['correct'], 'prediction')}")
        print(f"Incorrect setting prediction in sublayers:\n"
              f"{get_sublayer_counts(layers[model]['incorrect'], 'prediction')}")
        plot_min_entity_description_layers(pd.concat([layers[model]['correct'], layers[model]["incorrect"]]),
                                           f"{plots_path}/min_entity_description_layers.pdf",
                                           layer_count[model])

    entity_descriptions = {}
    for model in args.models:
        print(f"Processing {model} entity descriptions")
        dataset_path = f"{args.datasets_dir}/{model}"
        plots_path = f"{args.output_path}/{model}"
        entity_descriptions[model] = {
            "e1": load_entity_description(f"{dataset_path}/composition_correct/entity_description", "e1"),
            "last": load_entity_description(f"{dataset_path}/composition_correct/entity_description", "last")
        }
        print(f"Correct setting entity description counts:\n{get_entity_counts(layers[model]['correct'])}")
        print(f"Incorrect setting entity description counts:\n{get_entity_counts(layers[model]['incorrect'])}")
        plot_layer_matrix(entity_descriptions[model]["e1"], "e2", f"{plots_path}/entity_description_e1_e2.pdf")
        plot_layer_matrix(entity_descriptions[model]["last"], "e2", f"{plots_path}/entity_description_last_e2.pdf")
        plot_layer_matrix(entity_descriptions[model]["last"], "e3", f"{plots_path}/entity_description_last_e3.pdf")

    backpatching = {}
    for model in args.models:
        print(f"Processing {model} backpatching")
        dataset_path = f"{args.datasets_dir}/{model}"
        plots_path = f"{args.output_path}/{model}"
        backpatching[model] = {
            "correct": {
                "e1": load_backpatching(f"{dataset_path}/composition_correct/activation_patching/e1_original_activation_patching_classified.csv"),
                "last": load_backpatching(f"{dataset_path}/composition_correct/activation_patching/last_original_activation_patching_classified.csv")
            },
            "incorrect": {
                "e1": load_backpatching(f"{dataset_path}/composition_incorrect/activation_patching/e1_original_activation_patching_classified.csv"),
                "last": load_backpatching(f"{dataset_path}/composition_incorrect/activation_patching/last_original_activation_patching_classified.csv")
            }
        }
        plot_layer_matrix(backpatching[model]["incorrect"]["e1"], True, f"{plots_path}/backpatching_e1.pdf")
        plot_layer_matrix(backpatching[model]["incorrect"]["last"], True, f"{plots_path}/backpatching_last.pdf")
        print(f"Backpatching success:")
        print(calc_backpatching_success(layers[model]))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("datasets_dir")
    parser.add_argument("output_path")
    parser.add_argument("-m", "--models", nargs="+", required=True, choices=[
        "meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-13b-hf",
        "meta-llama/Meta-Llama-3-8B", "meta-llama/Meta-Llama-3-70B"])
    main(parser.parse_args())
