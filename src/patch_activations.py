from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path

import pandas as pd

from datasets import Dataset
from patched_generation import get_hidden_states, generate_with_patching_all_layers
from utils import get_layer_names, load_model, load_tokenizer, last_relation_word


def activation_patching(entries, model, tokenizer, source, target, default_decoding):
    target_token_str = "x"
    if source == "e1":
        source_str = [t.format(e) for t, e in zip(entries["r1_template"], entries["e1_label"])]
        if target == "original":
            target_token_str = entries["e1_label"]
    elif source == "r1":
        source_str = [last_relation_word(r1) for r1 in entries["r1_template"]]
    elif source == "r2":
        source_str = [last_relation_word(r2) for r2 in entries["r2_template"]]
    elif source == "last":
        source_str = [f"{t.format(e)} is" for t, e in zip(entries["r1_template"], entries["e1_label"])]
        if target == "original":
            target_token_str = "is"
    else:
        raise ValueError(f"Source {source} not supported")

    tokenizer.padding_side = "right"
    hidden_states = get_hidden_states(model, tokenizer, source_str, entries["source_prompt"])
    tokenizer.padding_side = "left"
    generations = generate_with_patching_all_layers(model, tokenizer, hidden_states, entries["target_prompt"],
                                                    target_token_str, default_decoding)

    entry_count = len(entries["source_prompt"])
    layer_count = len(get_layer_names(model))
    new_entries = {}
    for k in entries.keys():
        new_entries[k] = []
    new_entries["source_layer"] = []
    new_entries["target_layer"] = []
    new_entries["generation"] = []
    for source_layer in range(layer_count):
        for target_layer in range(layer_count):
            for i in range(entry_count):
                for k, v in entries.items():
                    new_entries[k].append(v[i])
                new_entries["source_layer"].append(source_layer)
                new_entries["target_layer"].append(target_layer)
                new_entries["generation"].append(generations[source_layer, target_layer][i])

    return new_entries


def create_target_prompt(dataset, source, target):
    if source == "e1":
        if target == "composition":
            return dataset.apply(lambda row: row["source_prompt"].replace(row["e1_label"], "x"), axis=1)
        elif target == "second-hop":
            return dataset.apply(lambda row: f"{row['r2_template'].format('x')} is", axis=1)
        elif target == "original":
            return dataset["source_prompt"]
    elif source == "r2":
        if target == "composition":
            return dataset.apply(lambda row: f"the x of {row['r1_template'].format(row['e1_label'])} is", axis=1)
    elif source == "last":
        return dataset["source_prompt"]
    else:
        raise ValueError(f"Source {source} and target {target} combination not supported")


def main(args):
    print(args)
    if not args.input_path:
        args.input_path = f"datasets/{args.model_name}/two_hop.csv"
    dataset = pd.read_csv(args.input_path, index_col=0)
    dataset["target_prompt"] = create_target_prompt(dataset, args.source, args.target)
    dataset = Dataset.from_pandas(dataset, preserve_index=False)

    model = load_model(args.model_name)
    model.eval()
    tokenizer = load_tokenizer(args.model_name)
    generations = dataset.map(
        activation_patching,
        fn_kwargs={"model": model, "tokenizer": tokenizer, "source": args.source, "target": args.target,
                   "default_decoding": args.default_decoding},
        batched=True,
        batch_size=args.batch_size,
        remove_columns=dataset.column_names
    )

    generations = generations.to_pandas()
    generations = generations.sort_values(["id", "source_layer", "target_layer"])
    generations = generations.reset_index(drop=True)
    if not args.output_path:
        args.output_path = f"datasets/{args.model_name}/activation_patching/{args.source}_{args.target}_activation_patching.csv"
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generations.to_csv(output_path, escapechar='\\')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("model_name", choices=[
        "gpt2",
        "meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-13b-hf",
        "meta-llama/Meta-Llama-3-8B", "meta-llama/Meta-Llama-3-70B"])
    parser.add_argument("source", choices=["e1", "r1", "r2", "last"])
    parser.add_argument("target", choices=["original", "composition", "second-hop"])
    parser.add_argument("--input-path")
    parser.add_argument("--output-path")
    parser.add_argument("--default-decoding", action=BooleanOptionalAction, default=False)
    parser.add_argument("--batch-size", type=int, default=32)
    main(parser.parse_args())
