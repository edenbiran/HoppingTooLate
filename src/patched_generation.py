import functools

import numpy as np
import torch
import torch.nn.functional as F
from baukit import Trace, TraceDict

from utils import get_prepend_space, get_layer_names, get_attention_modules, get_attention_layers_names, \
    get_norm_module, get_mlp_layers_names


def find_tokens(tokenizer, string_tokens, substring, prepend_space, last=True):
    if prepend_space:
        substring = " " + substring
    substring_tokens = tokenizer(substring, add_special_tokens=False, return_tensors="pt").input_ids[0]
    substring_tokens = substring_tokens.to(string_tokens.device)
    for start in range(len(string_tokens) - len(substring_tokens) + 1):
        end = start + len(substring_tokens)
        if torch.all(string_tokens[start:end] == substring_tokens):
            if last:
                return end - 1
            else:
                return start, end - 1
    return None


def get_hidden_states(model, tokenizer, entity, prompt):
    prompt_inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    prepend_space = get_prepend_space(model)
    entity_idx = [find_tokens(tokenizer, q, e, prepend_space) for q, e in zip(prompt_inputs.input_ids, entity)]
    entity_hidden_states = [None for _ in range(len(entity_idx))]

    layers = get_layer_names(model)
    with torch.no_grad(), TraceDict(model, layers) as trace:
        prompt_inputs = prompt_inputs.to(model.device)
        model(**prompt_inputs)
        for i, idx in enumerate(entity_idx):
            if idx is not None:
                entity_hidden_states[i] = torch.stack([trace[layer].output[0][i][idx].cpu() for layer in layers])

    return torch.stack(entity_hidden_states)


def get_hidden_states_with_patching(model, tokenizer, entity, prompt, hidden_state, target_layer):
    prompt_inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    prepend_space = get_prepend_space(model)
    entity_idx = [find_tokens(tokenizer, q, e, prepend_space) for q, e in zip(prompt_inputs.input_ids, entity)]
    entity_hidden_states = [None for _ in range(len(entity_idx))]

    def replace_hidden_state_hook(output):
        hs = output[0]
        if hs.shape[1] == 1:  # After first replacement the hidden state is cached
            return output
        hs[entity_idx] = hidden_state.to(hs.device)
        return (hs,) + output[1:]

    layers = get_layer_names(model)
    with torch.no_grad(), Trace(model, layer=target_layer, edit_output=replace_hidden_state_hook), TraceDict(model,
                                                                                                             layers) as trace:
        prompt_inputs = prompt_inputs.to(model.device)
        model(**prompt_inputs)
        for i, idx in enumerate(entity_idx):
            if idx is not None:
                entity_hidden_states[i] = torch.stack([trace[layer].output[0][i][idx].cpu() for layer in layers])

    return torch.stack(entity_hidden_states)


def generate_patching_inputs(model, tokenizer, target_prompt, target_token_str):
    if get_prepend_space(model):
        if type(target_token_str) is str:
            target_token_str = " " + target_token_str
        else:
            target_token_str = [" " + t for t in target_token_str]
    target_token = tokenizer(target_token_str, return_tensors="pt", add_special_tokens=False,
                             padding=True).input_ids[..., -1]
    inputs = tokenizer(target_prompt, return_tensors="pt", padding=True)
    target_position = (inputs.input_ids == target_token.unsqueeze(1)).cumsum(dim=1).argmax(dim=1)
    return inputs.to(model.device), (torch.LongTensor(range(target_position.shape[0])), target_position)


def decode_generated(tokenizer, generated, target_prompt, sampled=False):
    text = tokenizer.batch_decode(generated, skip_special_tokens=True)
    batch_size = len(target_prompt)
    sample_size = len(text) // len(target_prompt)
    if sampled:
        target_prompt = np.repeat(target_prompt, sample_size)
    pred = [t[len(p):].strip().replace("\n", " ") for t, p in zip(text, target_prompt)]
    if sampled:
        pred = np.split(np.array(pred), batch_size)
    return pred


def generate_with_patching_layer(model, tokenizer, hidden_state, target_layer, target_prompt, target_token_str,
                                 do_sample):
    inputs, target_position = generate_patching_inputs(model, tokenizer, target_prompt, target_token_str)

    def replace_hidden_state_hook(output):
        hs = output[0]
        if hs.shape[1] == 1:  # After first replacement the hidden state is cached
            return output
        hs[target_position] = hidden_state.to(hs.device)
        return (hs,) + output[1:]

    with torch.no_grad(), Trace(model, layer=target_layer, retain_output=False, edit_output=replace_hidden_state_hook):
        if do_sample:
            generated = model.generate(**inputs, pad_token_id=tokenizer.eos_token_id, max_new_tokens=10, do_sample=True)
        else:
            generated = model.generate(**inputs, do_sample=False, temperature=1, top_p=1, num_beams=1,
                                       pad_token_id=tokenizer.eos_token_id, max_new_tokens=10)

    return decode_generated(tokenizer, generated, target_prompt)


def generate_with_patching_same_layers(model, tokenizer, hidden_states, target_prompt, target_token_str, do_sample):
    generations_by_layer = []
    layer_names = get_layer_names(model)
    for source_layer_idx, target_layer in enumerate(layer_names):
        hidden_state = hidden_states[:, source_layer_idx, :]
        generations = generate_with_patching_layer(model, tokenizer, hidden_state, target_layer, target_prompt,
                                                   target_token_str, do_sample)
        generations_by_layer.append(generations)
    return generations_by_layer


def generate_with_patching_all_layers(model, tokenizer, hidden_states, target_prompt, target_token_str, do_sample):
    layer_names = get_layer_names(model)
    layer_count = len(get_layer_names(model))
    generations = np.ndarray((layer_count, layer_count), dtype=object)
    for source_layer in range(layer_count):
        for target_layer in range(layer_count):
            hidden_state = hidden_states[:, source_layer, :]
            generations[source_layer, target_layer] = generate_with_patching_layer(model,
                                                                                   tokenizer,
                                                                                   hidden_state,
                                                                                   layer_names[target_layer],
                                                                                   target_prompt,
                                                                                   target_token_str,
                                                                                   do_sample)
    return generations


def get_top_k_tokens(tokenizer, scores, k=10):
    with torch.no_grad():
        probabilities = F.softmax(scores, dim=-1)
        top_k_probabilities, top_k_indices = torch.topk(probabilities, k, dim=-1)
        top_k_words = np.empty_like(top_k_indices.cpu(), dtype=object)
        for b, batch in enumerate(top_k_indices):
            for l, layer in enumerate(batch):
                top_k_words[b][l] = tokenizer.batch_decode(layer.unsqueeze(-1))
        return top_k_words, top_k_probabilities


def get_token_ranks(projections, tokens):
    projections = projections.float().cpu()
    vals = projections[:, tokens]
    ranks = np.empty_like(vals, dtype=int)
    for layer in range(projections.shape[0]):
        for i, val in enumerate(vals[layer]):
            ranks[layer][i] = (projections[layer] > val).sum()
    return ranks


def generate_with_attention_knockout(model, tokenizer, layer_idx, prompt, source, target_token_str, default_decoding,
                                     k=0, return_probabilities=False):
    inputs, target_position = generate_patching_inputs(model, tokenizer, prompt, target_token_str)
    if type(source) is not str:
        prepend_space = get_prepend_space(model)
        source_position = torch.LongTensor([find_tokens(tokenizer, q, e, prepend_space) for q, e in
                                            zip(inputs.input_ids, source)])

    def wrap_attention_forward(original_forward_func):
        @functools.wraps(original_forward_func)
        def knockout_attention_forward(*args, **kwargs):
            new_args = []
            new_kwargs = {}
            for arg in args:
                new_args.append(arg)
            for key, v in kwargs.items():
                new_kwargs[key] = v

            if "hidden_states" in kwargs:
                hidden_states = kwargs["hidden_states"]
            else:
                hidden_states = args[0]
            batch_size = hidden_states.shape[0]
            num_tokens = hidden_states.shape[1]
            attention_weight_size = (batch_size, model.config.num_attention_heads, num_tokens, num_tokens)
            prev_attention_mask = kwargs["attention_mask"]
            new_attention_mask = torch.zeros(attention_weight_size, dtype=prev_attention_mask.dtype).to(
                prev_attention_mask.device) + prev_attention_mask
            if source == "all":
                new_attention_mask[target_position[0], :, :, target_position[1]] = torch.finfo(model.dtype).min
            elif source == "last":
                if num_tokens != 1:
                    new_attention_mask[target_position[0], :, -1, target_position[1]] = torch.finfo(model.dtype).min
            else:
                if num_tokens != 1:
                    new_attention_mask[target_position[0], :, source_position, target_position[1]] = torch.finfo(
                        model.dtype).min

            new_kwargs["attention_mask"] = new_attention_mask
            return original_forward_func(*new_args, **new_kwargs)

        return knockout_attention_forward

    if layer_idx != -1:
        attention_modules = get_attention_modules(model, layer_idx, k)
        original_forward_funcs = [attention_modules.forward for attention_modules in attention_modules]
        for attention_module in attention_modules:
            attention_module.forward = wrap_attention_forward(attention_module.forward)

    with torch.no_grad():
        if default_decoding:
            generated = model.generate(**inputs, pad_token_id=tokenizer.eos_token_id, max_new_tokens=10,
                                       return_dict_in_generate=return_probabilities, output_scores=return_probabilities)
        else:
            generated = model.generate(**inputs, do_sample=False, temperature=1, top_p=1, num_beams=1,
                                       pad_token_id=tokenizer.eos_token_id, max_new_tokens=10,
                                       return_dict_in_generate=return_probabilities, output_scores=return_probabilities)

    if layer_idx != -1:
        for i, attention_module in enumerate(attention_modules):
            attention_module.forward = original_forward_funcs[i]

    if return_probabilities:
        generations = decode_generated(tokenizer, generated["sequences"], prompt)
        top_k_words, top_k_probabilities = get_top_k_tokens(tokenizer, generated["scores"][0])
        return generations, top_k_words, top_k_probabilities
    else:
        return decode_generated(tokenizer, generated, prompt)


def generate_with_attention_knockout_all_layers(model, tokenizer, prompt, source, target_token_str, default_decoding,
                                                k=0, return_probabilities=False):
    generations_by_layer = {}
    for layer in range(-1, model.config.num_hidden_layers):
        generations_by_layer[layer] = generate_with_attention_knockout(model, tokenizer, layer, prompt, source,
                                                                       target_token_str, default_decoding, k,
                                                                       return_probabilities)
    return generations_by_layer


def get_entity_ranks(model, tokenizer, projections, entity):
    entity = [ent + [e.lower() for e in ent] for ent in entity]
    if get_prepend_space(model):
        entity = [ent + [" " + e for e in ent] for ent in entity]
    tokenized_entity = [tokenizer(ent, add_special_tokens=False).input_ids for ent in entity]
    ranks = []
    for proj, tokens in zip(projections, tokenized_entity):
        first_tokens = [t[0] for t in tokens]
        ranks.append(get_token_ranks(proj, first_tokens))
    return ranks


def get_sublayer_projection(model, tokenizer, prompt, bridge_entities, answers, layers):
    prompt_inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)

    sublayer_values = []
    with torch.no_grad(), TraceDict(model, layers) as trace:
        logits = model(**prompt_inputs).logits
        for i in range(len(prompt)):
            values = []
            for layer in layers:
                if type(trace[layer].output) is tuple:
                    output = trace[layer].output[0]
                else:
                    output = trace[layer].output
                values.append(output[i][-1].cpu())
            sublayer_values.append(torch.stack(values))
    sublayer_values = torch.stack(sublayer_values)

    out_embeddings = model.get_output_embeddings()
    norm = get_norm_module(model)
    norm_device = next(norm.parameters()).device
    with torch.no_grad():
        projections = out_embeddings(norm(sublayer_values.to(norm_device)))
    top_k_words, top_k_probabilities = get_top_k_tokens(tokenizer, projections)

    answer_ranks = get_entity_ranks(model, tokenizer, projections, answers)
    bridge_entity_ranks = get_entity_ranks(model, tokenizer, projections, bridge_entities)

    prediction_ranks = []
    predicted_tokens = logits[:, -1, :].max(dim=-1).indices
    for proj, predicted_token in zip(projections, predicted_tokens):
        prediction_ranks.append(get_token_ranks(proj, [predicted_token]))
    prediction_ranks = [r.flatten().tolist() for r in prediction_ranks]

    return top_k_words, top_k_probabilities, bridge_entity_ranks, answer_ranks, prediction_ranks


def get_attention_projection(model, tokenizer, prompt, bridge_entities, answers):
    return get_sublayer_projection(model, tokenizer, prompt, bridge_entities, answers,
                                   get_attention_layers_names(model))


def get_mlp_projection(model, tokenizer, prompt, bridge_entities, answers):
    return get_sublayer_projection(model, tokenizer, prompt, bridge_entities, answers, get_mlp_layers_names(model))
