from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

from datasets import Dataset
from utils import load_model, load_tokenizer, generate_and_test_answers, get_answers


def eval_composition(entries, model, tokenizer):
    return {
        "composition_correct": generate_and_test_answers(model, tokenizer, entries["source_prompt"],
                                                         entries["e3_answers"])
    }


def eval_first_hop(entries, model, tokenizer):
    prompts = [f"{r1_template.format(e1_label)} is" for r1_template, e1_label in
               zip(entries["r1_template"], entries["e1_label"])]
    return {
        "first_hop_correct": generate_and_test_answers(model, tokenizer, prompts, entries["e2_answers"])
    }


def eval_second_hop(entries, model, tokenizer):
    prompts = [f"{r2_template.format(e2_label)} is" for r2_template, e2_label in
               zip(entries["r2_template"], entries["e2_label"])]
    return {
        "second_hop_correct": generate_and_test_answers(model, tokenizer, prompts, entries["e3_answers"])
    }


def eval_entity_shortcut(entries, model, tokenizer):
    prompts = [f"{r2_template.format(e1_label)} is" for r2_template, e1_label in
               zip(entries["r2_template"], entries["e1_label"])]
    return {
        "entity_shortcut_correct": generate_and_test_answers(model, tokenizer, prompts, entries["e3_answers"])
    }


def eval_relation_shortcut(entries, model, tokenizer):
    prompts = [f"{r2_template.format(r1_template.format(''))}"[:-4] for r2_template, r1_template in
               zip(entries["r2_template"], entries["r1_template"])]
    return {
        "relation_shortcut_correct": generate_and_test_answers(model, tokenizer, prompts, entries["e3_answers"])
    }


def main(args):
    print(args)
    if not args.input_path:
        args.input_path = "../datasets/two_hop.csv"
    dataset = pd.read_csv(args.input_path, index_col=0)
    dataset = Dataset.from_pandas(dataset, preserve_index=False)
    dataset = dataset.map(get_answers, fn_kwargs={"key": "e2"})
    dataset = dataset.map(get_answers, fn_kwargs={"key": "e3"})

    model = load_model(args.model_name)
    model.eval()
    tokenizer = load_tokenizer(args.model_name)
    tokenizer.padding_side = "left"

    dataset = dataset.map(eval_first_hop, batched=True, batch_size=args.batch_size,
                          fn_kwargs={"model": model, "tokenizer": tokenizer})
    dataset = dataset.map(eval_second_hop, batched=True, batch_size=args.batch_size,
                          fn_kwargs={"model": model, "tokenizer": tokenizer})
    dataset = dataset.map(eval_composition, batched=True, batch_size=args.batch_size,
                          fn_kwargs={"model": model, "tokenizer": tokenizer})
    dataset = dataset.map(eval_entity_shortcut, batched=True, batch_size=args.batch_size,
                          fn_kwargs={"model": model, "tokenizer": tokenizer})
    dataset = dataset.map(eval_relation_shortcut, batched=True, batch_size=args.batch_size,
                          fn_kwargs={"model": model, "tokenizer": tokenizer})

    dataset_path = Path(f"datasets/{args.model_name}/two_hop.csv")
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = dataset.to_pandas()
    dataset = dataset.drop(["e2_answers", "e3_answers"], axis=1)
    dataset = dataset.reset_index(drop=True)

    if not args.output_path:
        args.output_path = f"datasets/{args.model_name}/two_hop.csv"
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("model_name", choices=[
        "gpt2",
        "meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-13b-hf",
        "meta-llama/Meta-Llama-3-8B", "meta-llama/Meta-Llama-3-70B"])
    parser.add_argument("--input-path")
    parser.add_argument("--output-path")
    parser.add_argument("--batch-size", type=int, default=32)
    main(parser.parse_args())
