import torch
import torch.nn.functional as F
import numpy as np
from einops import rearrange

from itertools import chain


def extract_heat_maps(controller, heatmap_dim):
  '''Extract heat maps from controller object'''
  return torch.cat([
    F.interpolate(rearrange(maps.cpu(), 'b (h w) t -> b t h w', h=int(np.sqrt(maps.shape[1]))), size=(heatmap_dim, heatmap_dim), mode='bicubic').clip(min=0)
    for maps
    in list(chain.from_iterable([maps for attn_type, maps in controller.attention_store.items() if 'cross' in attn_type]))
  ]).mean(axis=0).clip(max=1)


# https://github.com/castorini/daam/blob/main/daam/trace.py
def compute_token_merge_indices(tokenizer, prompt: str, word: str, word_idx: int = None, offset_idx: int = 0):
    merge_idxs = []
    tokens = tokenizer.tokenize(prompt.lower())
    if word_idx is None:
        word = word.lower()
        search_tokens = tokenizer.tokenize(word)
        start_indices = [x + offset_idx for x in range(len(tokens)) if tokens[x:x + len(search_tokens)] == search_tokens]
        for indice in start_indices:
            merge_idxs += [i + indice for i in range(0, len(search_tokens))]
        if not merge_idxs:
            raise ValueError(f'Search word {word} not found in prompt {prompt}!')
    else:
        merge_idxs.append(word_idx)

    return [x + 1 for x in merge_idxs], word_idx  # Offset by 1.
