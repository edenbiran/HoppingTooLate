from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

from datasets import Dataset
from patched_generation import get_attention_projection, get_mlp_projection
from utils import load_model, load_tokenizer, get_answers


def project_attention(entries, model, tokenizer, target):
    tokenizer.padding_side = "left"
    if target == "attention":
        top_k_words, top_k_probs, bridge_entity_ranks, answer_ranks, prediction_ranks = (
            get_attention_projection(model,
                                     tokenizer,
                                     entries["source_prompt"],
                                     entries["e2_answers"],
                                     entries["e3_answers"]))
    elif target == "mlp":
        top_k_words, top_k_probs, bridge_entity_ranks, answer_ranks, prediction_ranks = (
            get_mlp_projection(model,
                               tokenizer,
                               entries["source_prompt"],
                               entries["e2_answers"],
                               entries["e3_answers"]))
    else:
        raise ValueError(f"Target {target} not supported")

    new_entries = {}
    for k in entries.keys():
        new_entries[k] = []
    new_entries["layer"] = []
    new_entries["top_k_words"] = []
    new_entries["top_k_probs"] = []
    new_entries["e2_ranks"] = []
    new_entries["e3_ranks"] = []
    new_entries["prediction_rank"] = []
    for i in range(len(entries["source_prompt"])):
        for layer, (words, probs, bridge_ranks, ans_ranks, pred_rank) in enumerate(
                zip(top_k_words[i], top_k_probs[i], bridge_entity_ranks[i], answer_ranks[i], prediction_ranks[i])):
            for k, v in entries.items():
                new_entries[k].append(v[i])
            new_entries["layer"].append(layer)
            new_entries["top_k_words"].append(words.tolist())
            new_entries["top_k_probs"].append(probs.tolist())
            new_entries["e2_ranks"].append(bridge_ranks.tolist())
            new_entries["e3_ranks"].append(ans_ranks.tolist())
            new_entries["prediction_rank"].append(pred_rank)
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
    dataset = dataset.map(get_answers, fn_kwargs={"key": "e3"})
    dataset = dataset.map(get_answers, fn_kwargs={"key": "e2"})
    generations = dataset.map(
        project_attention,
        fn_kwargs={"model": model, "tokenizer": tokenizer, "target": args.target},
        batched=True,
        batch_size=args.batch_size,
        remove_columns=dataset.column_names
    )

    generations = generations.to_pandas()
    generations = generations.sort_values(["id", "layer"])
    generations = generations.reset_index(drop=True)
    generations["top_k_words"] = generations["top_k_words"].apply(list)
    generations["top_k_probs"] = generations["top_k_probs"].apply(list)
    generations["e2_ranks"] = generations["e2_ranks"].apply(list)
    generations["e3_ranks"] = generations["e3_ranks"].apply(list)
    if not args.output_path:
        args.output_path = f"datasets/{args.model_name}/sublayer_projection/{args.target}_projections.csv"
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generations.to_csv(output_path, escapechar='\\')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("model_name", choices=[
        "gpt2",
        "meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-13b-hf",
        "meta-llama/Meta-Llama-3-8B", "meta-llama/Meta-Llama-3-70B"])
    parser.add_argument("target", choices=["attention", "mlp"])
    parser.add_argument("--input-path")
    parser.add_argument("--output-path")
    parser.add_argument("--batch-size", type=int, default=32)
    main(parser.parse_args())
