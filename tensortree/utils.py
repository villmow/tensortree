from itertools import takewhile, repeat
import pathlib as pl
from typing import Union, List, Optional, Sequence, Any

import numpy as np
import torch

import functools


def validate_index(_func=None, allow_none: bool = False):
    """ Should be only used inside TensorTree and on functions that receive
    a node_idx as the first argument after self.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, node_idx, *args, **kwargs):

            if node_idx is None:
                if not allow_none:
                    raise IndexError(f"Index {node_idx} is not allowed in this method.")
            else:
                if node_idx < 0 or node_idx >= len(self):
                    raise IndexError(
                        f"Index {node_idx} is out of bounds for this"
                        f" {'sub' if self.is_subtree() else ''}tree with {len(self)} nodes."
                    )

            return func(self, node_idx, *args, **kwargs)
        return wrapper

    if _func is None:
        return decorator  # 2
    else:
        return decorator(_func)


def linecount(filename: Union[pl.Path, str]) -> int:
    with open(filename, 'rb') as f:
        bufgen = takewhile(
            lambda x: x,
            (
                f.raw.read(1024 * 1024) for _ in repeat(None)
            )
        )
        return sum(
            buf.count(b'\n') for buf in bufgen
        )


def get_project_root():
    return pl.Path(__file__).parent.parent


def to_torch(some_sequence: Sequence[Any]) -> torch.Tensor:
    if isinstance(some_sequence, torch.Tensor):
        return some_sequence
    elif isinstance(some_sequence, np.ndarray):
        return torch.from_numpy(some_sequence)
    else:
        return torch.tensor(some_sequence)  # may raise additional errors


def is_tensor_type(sequence: Union[torch.Tensor, np.ndarray, Any]) -> bool:
    return isinstance(sequence, (torch.Tensor, np.ndarray))


def to_matmul_compatibility(x: torch.Tensor) -> torch.Tensor:
    """ Brings a tensor into a matmul compatible dtype """
    if x.is_cuda:
        # bmm doesnt work with integers
        return x.float()
    elif x.dtype == torch.bool:
        # mm doesnt work on boolean arrays
        return x.long()
    elif not x.is_floating_point and (x.size(-1) >= torch.iinfo(x.dtype).max):
        # prevent overflow if we multiply uint8
        return x.long()

    return x


def apply_pad_mask_to_higher_dim_tensor(x: torch.Tensor, pad_idx: int, pad_mask: torch.Tensor):
    assert x.ndimension() == (pad_mask.ndimension() + 1), "x should have one dimension more than pad_mask"

    if pad_mask.ndimension() == 2:
        x[pad_mask] = pad_idx
        x[pad_mask[:, None, :].expand_as(x)] = pad_idx
    elif pad_mask.ndimension() == 1:
        # then x is 2-dimensional
        x[pad_mask] = pad_idx
        x[:, pad_mask] = pad_idx
    else:
        raise ValueError("pad mask should be either 1d or 2d")

    return x


def print_embed_overlap(embed_dict, vocab_dict):
    embed_keys = set(embed_dict.keys())
    vocab_keys = set(vocab_dict.symbols)
    overlap = len(embed_keys & vocab_keys)
    print("found {}/{} types in embedding file".format(overlap, len(vocab_dict)))


def parse_embedding(embed_path):
    """Parse embedding text file into a dictionary of word and embedding tensors.

    The first line can have vocabulary size and dimension. The following lines
    should contain word and embedding separated by spaces.

    Example:
        2 5
        the -0.0230 -0.0264  0.0287  0.0171  0.1403
        at -0.0395 -0.1286  0.0275  0.0254 -0.0932
    """
    print("Loading embedding file from", embed_path)
    from tqdm import tqdm
    embed_dict = {}
    with open(embed_path) as f_embed:
        next(f_embed)  # skip header
        for line in tqdm(f_embed):
            pieces = line.rstrip().split(" ")
            embed_dict[pieces[0]] = torch.Tensor(
                [float(weight) for weight in pieces[1:]]
            )
    return embed_dict


def load_embedding(embed_dict, vocab, embedding):
    for idx in range(len(vocab)):
        token = vocab[idx]
        if token in embed_dict:
            embedding.weight.data[idx] = embed_dict[token]
    return embedding