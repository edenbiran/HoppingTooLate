import ast
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from utils import get_answers, check_answer_in_pred


def find_layers_by_classification(classified_generations, classification, col_name, apply_min):
    if "layer" in classified_generations.columns:
        layer_col = "layer"
    else:
        layer_col = "source_layer"
    source_layer_classifications = classified_generations.groupby(
        ["id", "classification"])[layer_col].apply(list).unstack("classification").reset_index()

    if apply_min:
        source_layer_classifications[classification] = source_layer_classifications[classification].apply(
            lambda x: [np.inf] if x is np.nan else x
        )
        source_layer_classifications[col_name] = source_layer_classifications[classification].apply(min)
        source_layer_classifications[col_name] = source_layer_classifications[col_name].replace(np.inf, np.nan)
    else:
        if classification not in source_layer_classifications.columns:
            source_layer_classifications[classification] = np.nan
        source_layer_classifications[col_name] = source_layer_classifications[classification].apply(
            lambda x: list(dict.fromkeys(x)) if type(x) is not float else x
        )

    return source_layer_classifications[["id", col_name]]


def classify_by_vanilla(generations, merged):
    merged["answers"] = merged.apply(lambda row: get_answers(row, "e3")["e3_answers"], axis=1)
    merged["generation_correct"] = merged.apply(lambda row: check_answer_in_pred(row["generation"], row["answers"]),
                                                axis=1)
    merged["generation_vanilla_correct"] = merged.apply(
        lambda row: check_answer_in_pred(row["generation_vanilla"], row["answers"]), axis=1)
    return ((merged["generation_vanilla_correct"] & merged["generation_correct"]) |
            (generations["generation"] == merged["generation_vanilla"]))


def classify_attention_knockout(generations, target=None):
    vanilla_generations = generations[generations["layer"] == -1]
    merged = pd.merge(generations, vanilla_generations, on=["id"], suffixes=("", "_vanilla"))
    return classify_by_vanilla(generations, merged)


def classify_sublayer_projection(generations, target):
    if generations[target].dtype == np.int64:
        return generations[target].apply(lambda x: x == 0)
    else:
        return generations[target].apply(lambda x: any(r == 0 for r in ast.literal_eval(x)))


def classify_prediction_correct_row(row):
    if pd.isna(row["generation"]):
        return False
    generation = row["generation"]
    e3_answers = get_answers(row, "e3")["e3_answers"]
    if check_answer_in_pred(generation, e3_answers):
        return True
    return False


def classify_prediction_correct(generations, target=None):
    return generations.apply(classify_prediction_correct_row, axis=1)


def get_entity_classification_answers(row, key):
    return ([f": {a} " for a in get_answers(row, key)[f"{key}_answers"]] +
            [f": {a}," for a in get_answers(row, key)[f"{key}_answers"]] +
            [f": {a}." for a in get_answers(row, key)[f"{key}_answers"]] +
            [f": {a}:" for a in get_answers(row, key)[f"{key}_answers"]] +
            [f"{a}: " for a in get_answers(row, key)[f"{key}_answers"]])


def classify_entity_row(row):
    if pd.isna(row["generation"]):
        return "other"
    generation = row["generation"]
    e1_answers = get_entity_classification_answers(row, "e1")
    e2_answers = get_entity_classification_answers(row, "e2")
    e3_answers = get_entity_classification_answers(row, "e3")
    if check_answer_in_pred(generation, e1_answers):
        return "e1"
    elif check_answer_in_pred(generation, e2_answers):
        return "e2"
    elif check_answer_in_pred(generation, e3_answers):
        return "e3"
    else:
        return "other"


def classify_entity(generations, target=None):
    return generations.apply(classify_entity_row, axis=1)


def classify_generations(generations_path, classification_fn, target=None, prev_layers=False, use_cache=True):
    generations_path = Path(generations_path)
    classified_generations_path = generations_path.parent / f"{generations_path.stem}_classified.csv"

    if use_cache and classified_generations_path.exists():
        generations = pd.read_csv(classified_generations_path, index_col=0)
        print(f"Loaded classified generations from {classified_generations_path}")

    else:
        try:
            generations = pd.read_csv(generations_path, index_col=0, dtype={"id": int})
        except ValueError:
            print(f"Fixing generations from {generations_path}")
            with open(generations_path, "rb") as f:
                content = f.read()
            content = content.replace(b"\xe3\x80\x82", b" ")
            content = content.replace(b"\x0d", b"")
            with open(generations_path, "wb") as f:
                f.write(content)
            generations = pd.read_csv(generations_path, index_col=0, dtype={"id": int})

        print(f"Loaded generations from {generations_path}")
        generations["classification"] = classification_fn(generations, target)
        print(f"Saved classified generations to {classified_generations_path}")
        generations.to_csv(classified_generations_path)

    if "layer" in generations.columns:
        generations = generations[generations["layer"] != -1]
    if prev_layers:
        generations.loc[generations["source_layer"] <= generations["target_layer"], "classification"] = "masked"
    return generations


def load_attention_knockouts(dataset_dir):
    ret = defaultdict(dict)
    for source in ["last", "e1", "r1", "r2"]:
        for target in ["e1", "r1", "r2"]:
            path = f"{dataset_dir}/attention_knockout/{source}_{target}_attention_knockout_k3_composition.csv"
            if Path(path).exists():
                generations = classify_generations(path, classify_attention_knockout)
                ret[source][target] = find_layers_by_classification(
                    generations,
                    False,
                    f"{source}_{target}_attention_knockout_layers",
                    False)
    return ret


def load_activation_patching(dataset_dir):
    generations = defaultdict(dict)
    layers = defaultdict(dict)
    for source in ["e1", "r1", "r2", "last"]:
        for target in ["composition", "second-hop", "original"]:
            path = f"{dataset_dir}/activation_patching/{source}_{target}_activation_patching.csv"
            if Path(path).exists():
                generations[source][target] = classify_generations(path, classify_prediction_correct, prev_layers=True)
                layers[source][target] = find_layers_by_classification(generations[source][target],
                                                                       True,
                                                                       f"{source}_{target}_activation_patching_layer",
                                                                       False)
    return layers


def load_entity_description_generations(dataset_dir):
    generations = {}
    layers = defaultdict(dict)
    for source in ["last", "e1"]:
        gens = []
        for seed in range(3):
            path = f"{dataset_dir}/entity_description/{source}_entity_description_s{seed}.csv"
            gens.append(classify_generations(path, classify_entity))
        generations[source] = pd.concat(gens).drop_duplicates(
            subset=["id", "source_layer", "target_layer", "classification"]).sort_values(
            ["id", "source_layer", "target_layer"])
        for target in ["e2", "e3"]:
            layers[source][target] = find_layers_by_classification(generations[source],
                                                                   target,
                                                                   f"{source}_{target}_entity_layer",
                                                                   False)
    return layers


def load_sublayer_projections(dataset_dir):
    layers = defaultdict(dict)
    for sublayer in ["attention", "mlp"]:
        path = f"{dataset_dir}/sublayer_projection/{sublayer}_projections.csv"
        if Path(path).exists():
            generations = classify_generations(path, classify_sublayer_projection, target="prediction_rank",
                                               use_cache=False)
            layers[sublayer]["prediction"] = find_layers_by_classification(generations,
                                                                           True,
                                                                           f"{sublayer}_prediction_projection_layers",
                                                                           False)
            generations = classify_generations(path, classify_sublayer_projection, target="e2_ranks", use_cache=False)
            layers[sublayer]["e2"] = find_layers_by_classification(generations,
                                                                   True,
                                                                   f"{sublayer}_e2_projection_layers",
                                                                   False)
    return layers


def main(args):
    entity_description_layers = load_entity_description_generations(args.results_dir)
    attention_knockout_layers = load_attention_knockouts(args.results_dir)
    sublayer_projection_layers = load_sublayer_projections(args.results_dir)
    activation_patching_layers = load_activation_patching(args.results_dir)

    two_hop = pd.read_csv(f"{args.results_dir}/two_hop.csv", index_col=0)
    base = two_hop[["id", "composition_correct"]]
    df = base.merge(
        entity_description_layers["e1"]["e2"], on="id", how="left").merge(
        entity_description_layers["e1"]["e3"], on="id", how="left").merge(
        entity_description_layers["last"]["e2"], on="id", how="left").merge(
        entity_description_layers["last"]["e3"], on="id", how="left").merge(
        activation_patching_layers["e1"]["original"], on="id", how="left").merge(
        activation_patching_layers["last"]["original"], on="id", how="left").merge(
        sublayer_projection_layers["attention"]["prediction"], on="id", how="left").merge(
        sublayer_projection_layers["attention"]["e2"], on="id", how="left").merge(
        sublayer_projection_layers["mlp"]["prediction"], on="id", how="left").merge(
        sublayer_projection_layers["mlp"]["e2"], on="id", how="left").merge(
        attention_knockout_layers["last"]["e1"], on="id", how="left")

    df.to_csv(f"{args.results_dir}/layers.csv")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("results_dir")
    main(parser.parse_args())
