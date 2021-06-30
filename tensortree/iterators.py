from typing import Optional, Generator, Union, Tuple

import torch

import tensortree


def mask_layer(node_incidences: torch.BoolTensor) -> Generator[torch.BoolTensor, None, None]:
    """
    Generates masks, which traverse each layer of the tree
    from bottom to top.
        1. Mask over all leaves.
        2. Mask over all parents of the leaves.
        3. .. parents of parents ...

    :param node_incidence_matrix:
    :return:
    """
    batched = node_incidences.ndim == 3
    if not batched:
        node_incidences = node_incidences[None, :, :]

    node_indicences = node_incidences.clone()

    while node_indicences.any():
        leaves = node_indicences.sum(-2) == 1
        yield leaves if batched else leaves.squeeze()
        node_indicences.masked_fill_(leaves[:, :, None], 0)


def mask_level(
    node_incidences, reverse=False, return_level: bool = False
) -> Generator[Union[torch.BoolTensor, Tuple[int, torch.BoolTensor]], None, None]:
    """
    Generates masks, which traverse each level of the tree
    from bottom to top.
        1. Mask over tokens with lowest level.
        ...

    :param node_incidence_matrix:
    :return:
    """
    all_levels = tensortree.levels(node_incidences)
    max_level = all_levels.max()
    for level in range(1, max_level + 1):
        level = level if reverse else (max_level + 1) - level
        mask = all_levels == level

        yield (level, mask,) if return_level else mask
