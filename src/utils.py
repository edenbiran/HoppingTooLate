import ast

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2LMHeadModel, \
    LlamaForCausalLM, GemmaForCausalLM


HF_TOKEN = ""


def load_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(model_name, device="cuda"):
    if "70b" in model_name or "70B" in model_name:
        return AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto",
                                                    token=HF_TOKEN)
    else:
        return AutoModelForCausalLM.from_pretrained(model_name, token=HF_TOKEN).to(device)


def get_layer_names(model):
    if type(model) is GPT2LMHeadModel:
        return [f"transformer.h.{i}" for i in range(model.config.num_hidden_layers)]
    elif type(model) is LlamaForCausalLM:
        return [f"model.layers.{i}" for i in range(model.config.num_hidden_layers)]
    elif type(model) is GemmaForCausalLM:
        return [f"model.layers.{i}" for i in range(model.config.num_hidden_layers)]
    else:
        raise ValueError(f"Model type {type(model)} not supported")


def get_attention_layers_names(model):
    if type(model) is GPT2LMHeadModel:
        return [f"transformer.h.{i}.attn" for i in range(model.config.num_hidden_layers)]
    elif type(model) is LlamaForCausalLM:
        return [f"model.layers.{i}.self_attn" for i in range(model.config.num_hidden_layers)]
    else:
        raise ValueError(f"Model type {type(model)} not supported")


def get_mlp_layers_names(model):
    if type(model) is GPT2LMHeadModel:
        return [f"transformer.h.{i}.mlp" for i in range(model.config.num_hidden_layers)]
    elif type(model) is LlamaForCausalLM:
        return [f"model.layers.{i}.mlp" for i in range(model.config.num_hidden_layers)]
    else:
        raise ValueError(f"Model type {type(model)} not supported")


def get_attention_modules(model, layer, k=0):
    bot = max(0, layer - k)
    top = min(layer + k + 1, model.config.num_hidden_layers)
    if type(model) is GPT2LMHeadModel:
        return [model.transformer.h[l].attn for l in range(bot, top)]
    elif type(model) is LlamaForCausalLM:
        return [model.model.layers[l].self_attn for l in range(bot, top)]
    elif type(model) is GemmaForCausalLM:
        return [model.model.layers[l].self_attn for l in range(bot, top)]
    else:
        raise ValueError(f"Model type {type(model)} not supported")


def get_norm_module(model):
    if type(model) is GPT2LMHeadModel:
        return model.transformer.ln_f
    elif type(model) is LlamaForCausalLM:
        return model.model.norm
    elif type(model) is GemmaForCausalLM:
        return model.model.norm
    else:
        raise ValueError(f"Model type {type(model)} not supported")


def get_prepend_space(model):
    if type(model) is GPT2LMHeadModel:
        return True
    elif type(model) is LlamaForCausalLM:
        if "Llama-3" in model.config._name_or_path:
            return True
        elif "Llama-2" in model.config._name_or_path:
            return False
    elif type(model) is GemmaForCausalLM:
        return True
    else:
        raise ValueError(f"Model type {type(model)} not supported")


def decode_generated(tokenizer, generated, prompts):
    text = tokenizer.batch_decode(generated, skip_special_tokens=True)
    pred = [t[len(p):].strip().replace("\n", " ") for t, p in zip(text, prompts)]
    return pred


def get_answers(entry, key):
    aliases = ast.literal_eval(entry[f"{key}_aliases"])
    aliases = [a for a in aliases if len(a) > 1]
    entity = entry[f"{key}_label"]
    return {
        f"{key}_answers": [entity] + aliases,
    }


def check_answer_in_pred(pred, answers):
    pred = pred.lower()
    return any([a.lower() in pred for a in answers])


def generate_and_test_answers(model, tokenizer, prompts, answers):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        generated = model.generate(**inputs, do_sample=False, temperature=1, top_p=1, num_beams=1,
                                   pad_token_id=tokenizer.eos_token_id, max_new_tokens=10)
    predictions = decode_generated(tokenizer, generated, prompts)
    for prompt, pred, a in zip(prompts, predictions, answers):
        print(f"Prompt: {prompt}\nPrediction: {pred}\nAnswers: {a}\nResults: {check_answer_in_pred(pred, a)}")
        print("-----------------------------------")
    return [check_answer_in_pred(p, a) for p, a in zip(predictions, answers)]


def print_dataset_statistics(dataset):
    print("Dataset statistics:")
    for target in ["e1", "r1", "e2", "r2", "e3"]:
        proportions = dataset[f"{target}_type"].value_counts(normalize=True) * 100
        counts = dataset[f"{target}_type"].value_counts()
        stats = pd.merge(counts, proportions, left_index=True, right_index=True)
        print(stats)


def rebalance_dataset(df, key="e2_type", size=100, secondary_key=None):
    if secondary_key is None:
        balanced = df.groupby(key).apply(lambda x: x.sample(min(size, len(x)))).reset_index(drop=True)
    else:
        balanced = df.groupby([key, secondary_key]).apply(lambda x: x.sample(min(size // 2, len(x)))).reset_index(drop=True)
    balanced = balanced.sort_values("id")
    return balanced


def last_relation_word(relation):
    return relation.split(" ")[-3]
