from itertools import takewhile, repeat
import pathlib as pl
import re
from typing import *

import torch
from torch.utils.data.dataloader import default_collate


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


def collate_tokens(
        values: List[torch.Tensor], pad_idx: int, left_pad: bool = False,
        pad_to_length: Optional[int] = None, pad_to_multiple: int = 1
):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)

    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size-0.1)//pad_to_multiple + 1) * pad_to_multiple)

    res = values[0].new_full((len(values), size), fill_value=pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
    return res


def collate_descendants(
        descendants: List[torch.Tensor], left_pad: bool = False, pad_idx: int = 0,
        pad_to_length: Optional[int] = None, pad_to_multiple: int = 1
):
    """Convert a list of 1d tensors of descendants into a padded 2d tensor. It will
    increment all values by (pad_idx + 1).
    """
    return collate_tokens(
        [v + (pad_idx + 1) for v in descendants],
        pad_idx, left_pad, pad_to_length, pad_to_multiple
    )


def collate_parents(
        parents: List[torch.Tensor], left_pad: bool = False, pad_idx: int = -2,
        pad_to_length: Optional[int] = None, pad_to_multiple: int = 1
):
    """Convert a list of 1d tensors of parent indices into a padded 2d tensor.
        Increments the tokens by the amount of added padding tokens.

        If pad_idx is part of the tensor, it will not be incremented!

        [ [-1, 0, 1, 0],
          [-1, 0, 1] ]
        ->
        [ [-1,  0, 1, 0],
          [-1, -1, 1, 2] ] if pad_idx = -1 and left_pad

     """
    size = max(v.size(0) for v in parents)
    size = size if pad_to_length is None else max(size, pad_to_length)

    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)

    res = parents[0].new_full((len(parents), size), fill_value=pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(parents):
        num_pad = size - len(v)
        if num_pad > 0:
            if left_pad:  # parent indices change if we left pad
                mask = (v != pad_idx)  # * (v != eos_idx)
                v[mask] += num_pad

        copy_tensor(v, res[i][num_pad:] if left_pad else res[i][:len(v)])

    return res


def collate_tree(
        batch, left_pad: bool = False, pad_idx: int = -1,
        pad_to_length: Optional[int] = None, pad_to_multiple: int = 1
):

    parents = collate_parents(
        [s["tree"].pop("parents") for s in batch],
        left_pad, pad_to_length=pad_to_length, pad_to_multiple=pad_to_multiple
    )
    descendants = collate_descendants(
        [s["tree"].pop("descendants") for s in batch],
        left_pad, pad_to_length=pad_to_length, pad_to_multiple=pad_to_multiple
    )
    tokens = collate_tokens(
        [s["tree"].pop("tokens") for s in batch],
        pad_idx, left_pad, pad_to_length, pad_to_multiple
    )
    labels = collate_descendants(
        [s["labels"] for s in batch],
        left_pad, pad_idx=-1, pad_to_length=pad_to_length, pad_to_multiple=pad_to_multiple
    )

    return {
        "labels": labels,  # process remaining items
        "tree": {
            "tokens": tokens,
            "descendants": descendants,
            "parents": parents,
        },
    }

