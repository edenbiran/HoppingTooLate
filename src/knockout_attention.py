from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path

import pandas as pd

from datasets import Dataset
from patched_generation import generate_with_attention_knockout_all_layers
from utils import load_model, load_tokenizer, last_relation_word


def generate_answer_with_attention_knockout(entries, model, tokenizer, default_decoding, knockout_source,
                                            knockout_target, k=0, prompt="composition"):
    if knockout_target == "e1":
        generation_target_token_string = [t.format(e) for t, e in zip(entries["r1_template"], entries["e1_label"])]
    elif knockout_target == "e2":
        generation_target_token_string = [t.format(e) for t, e in zip(entries["r2_template"], entries["e2_label"])]
    elif knockout_target == "r1":
        generation_target_token_string = [last_relation_word(r1) for r1 in entries["r1_template"]]
    elif knockout_target == "r2":
        generation_target_token_string = [last_relation_word(r2) for r2 in entries["r2_template"]]
    else:
        raise ValueError(f"Knockout target {knockout_target} not supported")

    if knockout_source == "e1":
        knockout_source = entries["e1_label"]
    elif knockout_source == "r1":
        knockout_source = [last_relation_word(r1) for r1 in entries["r1_template"]]

    if prompt == "composition":
        source_prompt = entries["source_prompt"]
    elif prompt == "first-hop":
        source_prompt = [f"{r1_template.format(e1_label)} is" for r1_template, e1_label in
                         zip(entries["r1_template"], entries["e1_label"])]
    elif prompt == "second-hop":
        source_prompt = [f"{r2_template.format(e2_label)} is" for r2_template, e2_label in
                         zip(entries["r2_template"], entries["e2_label"])]
    else:
        raise ValueError(f"Prompt {prompt} not supported")

    tokenizer.padding_side = "left"
    generations = generate_with_attention_knockout_all_layers(model, tokenizer, source_prompt,
                                                              knockout_source, generation_target_token_string,
                                                              default_decoding, k)

    new_entries = {}
    for k in entries.keys():
        new_entries[k] = []
    new_entries["layer"] = []
    new_entries["generation"] = []
    for layer in generations.keys():
        for i in range(len(entries["source_prompt"])):
            for k, v in entries.items():
                new_entries[k].append(v[i])
            new_entries["layer"].append(layer)
            generation = generations[layer]
            new_entries["generation"].append(generation[i])

    return new_entries


def main(args):
    print(args)
    if not args.input_path:
        args.input_path = f"datasets/{args.model_name}/two_hop.csv"
    dataset = pd.read_csv(args.input_path, index_col=0)
    dataset = Dataset.from_pandas(dataset, preserve_index=False)

    model = load_model(args.model_name)
    model.eval()
    tokenizer = load_tokenizer(args.model_name)
    generations = dataset.map(
        generate_answer_with_attention_knockout,
        fn_kwargs={"model": model, "tokenizer": tokenizer, "default_decoding": args.default_decoding,
                   "knockout_source": args.knockout_source, "knockout_target": args.knockout_target, "k": args.k,
                   "prompt": args.prompt},
        batched=True,
        batch_size=args.batch_size,
        remove_columns=dataset.column_names
    )

    generations = generations.to_pandas()
    generations = generations.sort_values(["id", "layer"])
    generations = generations.reset_index(drop=True)
    if not args.output_path:
        args.output_path = (f"datasets/"
                            f"{args.model_name}/"
                            f"attention_knockout/"
                            f"{args.knockout_source}_{args.knockout_target}_attention_knockout_k{args.k}_{args.prompt}.csv")
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generations.to_csv(output_path, escapechar='\\')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("model_name", choices=[
        "gpt2",
        "meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-13b-hf",
        "meta-llama/Meta-Llama-3-8B", "meta-llama/Meta-Llama-3-70B"])
    parser.add_argument("knockout_source", choices=["all", "last", "e1", "r1"])
    parser.add_argument("knockout_target", choices=["e1", "e2", "r1", "r2"])
    parser.add_argument("--input-path")
    parser.add_argument("--output-path")
    parser.add_argument('--default-decoding', action=BooleanOptionalAction, default=False)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--prompt", type=str, choices=["first-hop", "second-hop", "composition"],
                        default="composition")
    main(parser.parse_args())
