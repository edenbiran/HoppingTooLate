import random
from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from patched_generation import get_hidden_states, generate_with_patching_all_layers
from utils import get_layer_names, load_model, load_tokenizer


def generate_entity_description_with_patching(entries, model, tokenizer, source, do_sample):
    if source == "e1":
        entities = [t.format(e) for t, e in zip(entries["r1_template"], entries["e1_label"])]
    elif source == "last":
        entities = [prompt.split()[-1] for prompt in entries["source_prompt"]]
    else:
        raise ValueError(f"Source {source} not supported")

    tokenizer.padding_side = "right"
    hidden_states = get_hidden_states(model, tokenizer, entities, entries["source_prompt"])
    tokenizer.padding_side = "left"
    generations = generate_with_patching_all_layers(model, tokenizer, hidden_states, entries["target_prompt"],
                                                    "x", do_sample)

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


def set_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def main(args):
    print(args)
    set_seeds(args.seed)

    if not args.input_path:
        args.input_path = f"datasets/{args.model_name}/two_hop.csv"
    dataset = pd.read_csv(args.input_path, index_col=0)
    if args.target_prompt == "description":
        target_prompt = (
            "Syria: Syria is a country in the Middle East, " +
            "Leonardo DiCaprio: Leonardo DiCaprio is an American actor, " +
            "Samsung: Samsung is a South Korean multinational corporation, " +
            "x"
        )
    else:
        target_prompt = (
            "Syria: Syria, " +
            "Leonardo DiCaprio: Leonardo DiCaprio, " +
            "Samsung: Samsung, " +
            "x"
        )
    dataset["target_prompt"] = target_prompt
    dataset = Dataset.from_pandas(dataset, preserve_index=False)

    model = load_model(args.model_name)
    model.eval()
    tokenizer = load_tokenizer(args.model_name)
    generations = dataset.map(
        generate_entity_description_with_patching,
        fn_kwargs={"model": model, "tokenizer": tokenizer, "source": args.source,
                   "do_sample": args.do_sample},
        batched=True,
        batch_size=args.batch_size,
        remove_columns=dataset.column_names
    )

    generations = generations.to_pandas()
    generations = generations.sort_values(["id", "source_layer", "target_layer"])
    generations = generations.reset_index(drop=True)
    if not args.output_path:
        args.output_path = f"datasets/{args.model_name}/entity_description/{args.source}_entity_description_s{args.seed}.csv"
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generations.to_csv(output_path, escapechar='\\')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("model_name", choices=[
        "gpt2",
        "meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-13b-hf",
        "meta-llama/Meta-Llama-3-8B", "meta-llama/Meta-Llama-3-70B"])
    parser.add_argument("source", choices=["e1", "last"])
    parser.add_argument("--input-path")
    parser.add_argument("--output-path")
    parser.add_argument('--target-prompt', choices=["description", "repeat"], default="description")
    parser.add_argument('--do-sample', action=BooleanOptionalAction, default=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args())
