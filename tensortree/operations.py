from itertools import zip_longest
from typing import Union, Any, Optional, Sequence, List, Tuple

import numpy as np
import torch

import tensortree
from tensortree import TensorTree
from tensortree.utils import is_tensor_type, to_matmul_compatibility, apply_pad_mask_to_higher_dim_tensor


def node_incidence_matrix(
        descendants: torch.Tensor, pad_idx: int = -1, pad_mask: Optional[torch.Tensor] = None
) -> torch.BoolTensor:
    """
    Computes the node incidence matrix between nodes based on the descendants array.
    This array may be batched, then it is created for all trees at once.

     1D or 2D tensor of descendants [seqlen] or [bsz x seqlen].

    If the tensor has been padded, this function assumes descendants
     has been incremented by `pad_idx + 1` (to make sure pad_idx
     does not appear in the sequence).

    Example:
    >>> node_incidence_matrix(
    >>>    descendants=torch.tensor(
    >>>        [[6, 2, 4, 2, 2],
    >>>         [1, 5, 2, 3, 2],
    >>>         [1, 5, 2, 3, 2]]
    >>>    ),
    >>>    pad_idx=1
    >>> )

    OUTPUT:
    >>> tensor([[[ True, False, False, False, False],
    >>>          [ True,  True, False, False, False],
    >>>          [ True, False,  True, False, False],
    >>>          [ True, False,  True,  True, False],
    >>>          [ True, False,  True, False,  True]],
    >>>         [[False, False, False, False, False],
    >>>          [False,  True, False, False, False],
    >>>          [False,  True,  True, False, False],
    >>>          [False,  True, False,  True, False],
    >>>          [False,  True, False,  True,  True]],
    >>>         [[False, False, False, False, False],
    >>>          [False,  True, False, False, False],
    >>>          [False,  True,  True, False, False],
    >>>          [False,  True, False,  True, False],
    >>>          [False,  True, False,  True,  True]]])

    Every row contains the nodes that lay on the path to root
     for node at row. True if a node lays on the path to root, False otherwise.

    """
    assert descendants.ndimension() <= 2, "Wrong dimensions on descendants tensor"
    batched = descendants.ndimension() == 2
    B = descendants.size(0) if batched else 1
    S = descendants.size(-1)

    # make it work also with 1D tensors
    if not batched:
        descendants = descendants[None, :]

        if pad_mask is not None:
            pad_mask = pad_mask[None, :]  # unsqueeze also pad mask

    descendants = descendants.clone()

    # compute padding_masks before calculating the correct descendants, afterwards we can't distinguish
    pad_mask = (descendants == pad_idx) if pad_mask is None else pad_mask
    token_mask = ~pad_mask

    positions = torch.arange(end=S, device=descendants.device)
    reverse_range = torch.flip(positions, dims=(0,))

    positions = positions.expand_as(descendants)
    reverse_range = reverse_range.expand_as(descendants)

    # every node is on its own path to root
    descendants[token_mask] = descendants[token_mask] - (pad_idx + 1) + 1
    descendants[pad_mask] = reverse_range[pad_mask]

    reps = descendants.new_zeros((B, 3*S,))
    # num repetitions of pad before diag
    reps[:, ::3] = positions
    # num repetitions of position after diag
    reps[:, 1::3] = descendants
    # num repetitions of pad after diag
    reps[:, 2::3] = (descendants - S).abs() - positions

    val_mask = pad_mask.new_full(reps.shape, False)
    val_mask[:, 1::3] = token_mask
    mat = torch.repeat_interleave(
        val_mask.view(-1),
        reps.view(-1)
    ).view(B, S, S).transpose(1, 2)

    return mat if batched else mat.squeeze(0)


def adjacency_matrix(parents: torch.Tensor, directed: bool = True, direction_up: bool = False, loop_root: bool = False):
    """
    Generates the adjacency matrix for a parents array.

    Currently this works only for single arrays (so no batching).

    :param parents: Array of parents: `tree.parents`
    :param directed: Should the adjacency matrix be directed?
    :param direction_up: Per default the root points at its children.
    :param loop_root: Should root have a loop.
    :return:
    """
    root_val = parents[0].item()
    parents[0] = 0

    a = torch.arange(parents.size(-1), device=parents.device)
    adj = parents.new_zeros(parents.size(-1), parents.size(-1), dtype=torch.bool)
    if directed and direction_up:
        adj[a, parents] = True
    elif directed and not direction_up:
        adj[parents, a] = True
    else:
        adj[a, parents] = True
        adj[parents, a] = True

    parents[0] = root_val

    if not loop_root:
        adj[0, 0] = 0

    return adj


def incidences_to_nodes(node_incidences: torch.Tensor, pad_idx=-1):
    """
    This function turns the binary incidence_matrix into a long tensor
     containing the actual position of the node in the sequence.

    Example:
    >>> incidences = node_incidence_matrix(
    >>>    descendants=torch.tensor(
    >>>        [[6, 2, 4, 2, 2],
    >>>         [1, 5, 2, 3, 2],
    >>>         [1, 5, 2, 3, 2]]
    >>>    ),
    >>>    pad_idx=1
    >>> )

    >>> incidences
    >>> tensor([[[ True, False, False, False, False],
    >>>          [ True,  True, False, False, False],
    >>>          [ True, False,  True, False, False],
    >>>          [ True, False,  True,  True, False],
    >>>          [ True, False,  True, False,  True]],
    >>>         [[False, False, False, False, False],
    >>>          [False,  True, False, False, False],
    >>>          [False,  True,  True, False, False],
    >>>          [False,  True, False,  True, False],
    >>>          [False,  True, False,  True,  True]],
    >>>         [[False, False, False, False, False],
    >>>          [False,  True, False, False, False],
    >>>          [False,  True,  True, False, False],
    >>>          [False,  True, False,  True, False],
    >>>          [False,  True, False,  True,  True]]])

    OUTPUT:
    >>> incidences_to_nodes(incidences, pad_idx=-1)
    >>> tensor([[[ 0, -1, -1, -1, -1],
    >>>          [ 0,  1, -1, -1, -1],
    >>>          [ 0, -1,  2, -1, -1],
    >>>          [ 0, -1,  2,  3, -1],
    >>>          [ 0, -1,  2, -1,  4]],
    >>>         [[-1, -1, -1, -1, -1],
    >>>          [-1,  1, -1, -1, -1],
    >>>          [-1,  1,  2, -1, -1],
    >>>          [-1,  1, -1,  3, -1],
    >>>          [-1,  1, -1,  3,  4]],
    >>>         [[-1, -1, -1, -1, -1],
    >>>          [-1,  1, -1, -1, -1],
    >>>          [-1,  1,  2, -1, -1],
    >>>          [-1,  1, -1,  3, -1],
    >>>          [-1,  1, -1,  3,  4]]])


    :param node_incidences: Bool Tensor as returend from node_incidence_matrix
    :param pad_idx:
    :return:
    :rtype:
    """
    assert node_incidences.ndimension() <= 3, "Wrong dimensions on descendants tensor"
    batched = node_incidences.ndimension() == 3
    B = node_incidences.size(0) if batched else 1
    S = node_incidences.size(-1)

    if not batched:
        node_incidences = node_incidences[None,:,:]

    nodes = torch.arange(S).to(node_incidences.device, dtype=torch.long) # [S]
    nodes = nodes[None, None, :]  # 1, 1, S

    # node_incidences += (pad_idx + 1)
    nodes = nodes.expand_as(node_incidences).clone()  # [B, S, S]
    nodes[~node_incidences] = pad_idx

    if not batched:
        nodes = nodes.squeeze(0)

    return nodes


def levels(node_incidences: torch.Tensor) -> torch.LongTensor:
    """ Computes the level of each node based on a binary node incidence matrix.

    Root has level 0.

    For this tree:
        0
        ├──  1
        └──  2
            ├──  3
            └──  4
                └──  5

    With the following node incidence matrix:

    tensor([[1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0],
            [1, 0, 1, 1, 0, 0],
            [1, 0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1, 1]], dtype=torch.uint8)

    Will return:

    tensor([0, 1, 1, 2, 2, 3])
    """
    return node_incidences.sum(dim=-1) - 1

# def adjacency_matrix_batched(parents: torch.Tensor, directed: bool = False, direction_up: bool = False, pad_mask=None):
#     raise NotImplementedError("Wip")
#     batched = parents.ndim == 2
#     if not batched:
#         parents = parents[None, :]
#
#     B, S = parents.shape
#
#     if batched and pad_mask is None:
#         pad_mask = parents < -1
#
#     parents[:, 0], root_vals = 0, parents[:, 0]  # only if right padded
#
#     print("root_vals", root_vals)
#     print("->", parents)
#
#     a = torch.arange(S, device=parents.device)
#     b = torch.arange(B, device=parents.device)
#     adj = parents.new_zeros(B, S, S, dtype=torch.bool)
#
#     if directed and direction_up:
#         adj[a, parents] = True
#     elif directed and not direction_up:
#         adj[parents] = True
#     else:
#         adj[a,parents] = True
#         adj[parents, a] = True
#
#     parents[:, 0] = root_vals
#
#     return adj

def ancestral_matrix(node_incidences: torch.Tensor) -> torch.LongTensor:
    """
    Computes the level of the least common ancestor between any node pair
    (or in other words the length of the common prefix between two nodes).

    For this tree:
        0
        ├──  1
        └──  2
            ├──  3
            └──  4
                └──  5

    With the following node incidence matrix:

    tensor([[1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0],
            [1, 0, 1, 1, 0, 0],
            [1, 0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1, 1]], dtype=torch.uint8)

    This method will return:

    tensor([[0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 1],
            [0, 0, 1, 2, 1, 1],
            [0, 0, 1, 1, 2, 2],
            [0, 0, 1, 1, 2, 3]])

    Pytorch supports matrix multiplications on cuda tensors only with float tensors,
     thus this method will return a FloatTensor when the node incidence matrix is located
     on cuda. Otherwise it tries to keep the original dtype when possible, otherwise
     returns a LongTensor.

    :param node_incidences: [B, S, S] or [S, S] matrix of node incidences
    :return:
    """
    batched = node_incidences.ndimension() == 3
    node_incidences = to_matmul_compatibility(node_incidences)

    if batched:
        res = torch.bmm(
            node_incidences,
            node_incidences.transpose(1, 2)
        ) - 1
    else:
        res = (node_incidences @ node_incidences.t()) - 1

    return res.long()


def movements(node_incidences: torch.Tensor, pad_idx: int = -1, pad_mask: Optional[torch.Tensor] = None) -> torch.LongTensor:
    """
    Computes movements between nodes.

    For this tree:
        0
        ├──  1
        └──  2
            ├──  3
            └──  4
                └──  5

    With the following node incidence matrix:
    tensor([[1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0],
            [1, 0, 1, 1, 0, 0],
            [1, 0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1, 1]], dtype=torch.uint8)

    Will return the following movements between nodes.
    tensor([[0, 0, 0, 0, 0, 0],
            [1, 0, 1, 1, 1, 1],
            [1, 1, 0, 0, 0, 0],
            [2, 2, 1, 0, 1, 1],
            [2, 2, 1, 1, 0, 0],
            [3, 3, 2, 2, 1, 0]])

    You can read it the following:
    To walk from node i to node j one needs to make
     1. res[i, j] upward movements
     2. res[j, i] downward movements

    Some examples:
     - from 0 to 4 -> res[0,4]=0; res[4,0]=2 (0 steps up, 2 steps down)
     - from 1 to 5 -> res[1,5]=1; res[5,1]=3 (1 step  up, 3 steps down)

    Will always return a LongTensor.
    """
    batched = node_incidences.ndimension() == 3
    B = node_incidences.size(0) if batched else 1
    S = node_incidences.size(1)
    start = pad_idx + 1

    # on cuda we need to have a FloatTensor to do matmuls
    # so lets do that one time in advance
    if node_incidences.is_cuda:
        node_incidences = node_incidences.float()

    node_levels = levels(node_incidences)
    length_of_common_prefix = ancestral_matrix(node_incidences)

    if batched:
        moves = node_levels[:, None, :] - length_of_common_prefix
    else:
        moves = node_levels - length_of_common_prefix

    moves = moves.transpose(-2, -1)  #
    moves = moves.long()
    moves += start

    if pad_mask is not None:
        moves = apply_pad_mask_to_higher_dim_tensor(moves, pad_idx, pad_mask)

    return moves


def distances(
        node_incidences: torch.Tensor, pad_idx: int = -1, pad_mask: Optional[torch.Tensor] = None
) -> torch.LongTensor:
    """
    Calculates distances between every node and every other node in an undirected tree.

    Distances will begin at (pad_idx + 1), to support padded batches.
    """
    # batch size x sequence length
    batched = node_incidences.ndimension() == 3
    S = node_incidences.size(1)
    start = pad_idx + 1

    if not batched:
        node_incidences = node_incidences[None, :, :]  # add batch dim

    inverted_node_incidences = ~node_incidences

    # matmul on cuda works only with float tensors
    node_incidences = to_matmul_compatibility(node_incidences)
    inverted_node_incidences = to_matmul_compatibility(inverted_node_incidences)

    # we want to compute the amount of unequal elements between two pathes
    # (XOR between the pathes and afterwards sum the positive bits).
    distances = torch.bmm(node_incidences, inverted_node_incidences.transpose(1, 2))  # [B, S, S]
    distances += torch.bmm(inverted_node_incidences, node_incidences.transpose(1, 2))  # [B, S, S]

    distances = distances.long()
    distances += start

    if not batched:
        distances = distances.squeeze(0)  # remove batch dim

    if pad_mask is not None:
        distances = apply_pad_mask_to_higher_dim_tensor(
            distances,
            pad_idx,
            pad_mask
        )

    return distances


def least_common_ancestors(node_incidences: torch.Tensor) -> torch.LongTensor:
    """
    Computes least common ancestors for a 1D tensor of num descendants
    :param node_incidences:  shape [S, S]. Does not work when batched
    :return:
    """
    assert node_incidences.ndimension() == 2

    # FIXME/TODO  make it work with padded batches, currently only works with 1D
    S = node_incidences.size(-1)
    nodes = incidences_to_nodes(
        node_incidences,
        pad_idx=(-10 * S)  # Needs to be something small, otherwise wont work
    )

    # this works by comparing a path of a node with another path.
    # consider these two pathes:
    # [    0,     1,     2,     3, -1000, -1000, -1000, -1000],
    # [    0,     1,     2, -1000,     4, -1000, -1000, -1000]
    # if we add those together the LCA (2) will have the highest value.
    # [    0,     2,     4,  -997,  -996, -2000, -2000, -2000]
    #                    ^
    ancestors = torch.argmax(
        nodes[:, None, :] + nodes,  # [S, S, S]  expensive!
        dim=-1
    )  # [S, S]

    return ancestors


def delete_subtree(tree: TensorTree, node_idx: Union[int, torch.Tensor], replacement: Optional[Any] = None) -> TensorTree:
    """
    Returns a new tree with branch at node_idx deleted or replaced with a single node without children.

    Does the following, given this tree and node_idx=2, replacement_token=99:

    0 MethodDeclaration
    ├── 1 parameters
    │   ├── 2 FormalParameter
    │   │   ├── 3 type
    │   │   │   └── 4 ReferenceType
    │   │   │       └── 5 name
    │   │   │           └── 6 Bitmap
    │   │   └── 7 name
    │   │       └── 8 bmp
    │   └── 9 FormalParameter
    │       ├── 10 type
    │       │   └── 11 ReferenceType
    │       │       └── 12 name
    │       │           └── 13 File
    │       └── 14 name
    └── 15 body

    Will return tensors for the following tree:

    0 MethodDeclaration (0)
    ├── 1 parameters (1)
    │   ├── 2 <MASK> (2)
    │   └── 9 FormalParameter (3)
    │       ├── 10 type (4)
    │       │   └── 11 ReferenceType (5)
    │       │       └── 12 name (6)
    │       │           └── 13 File (7)
    │       └── 14 name (8)
    └── 15 body (9)

    The original tensors have been:
    tokens:        [  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15]
    parents:       [ -1,  0,  1,  2,  3,  4,  5,  2,  7,  1,  9, 10, 11, 12,  9,  0]
    #descendants:  [ 14, 13,  6,  3,  2,  1,  0,  1,  0,  5,  3,  2,  1,  0,  0,  0]
    #children:     [  2,  2,  2,  1,  1,  1,  0,  1,  0,  2,  1,  1,  1,  0,  0,  0]

    This method will return the following tensors:
    words:        [  0,  1, 99,  9, 10, 11, 12, 13, 14, 15]
    parents:      [ -1,  0,  1,  1,  3,  4,  5,  6,  3,  0]
    #descendants: [  9,  7,  0,  5,  3,  2,  1,  0,  0,  0]
    #children:    [  2,  2,  0,  2,  1,  1,  1,  0,  0,  0]
    """

    if replacement is not None:
        assert isinstance(replacement, (torch.Tensor, int, type(tree.node_data[node_idx]))), "Replacement token needs to be tensor type"

    delete_all_children = replacement is None

    # which nodes to keep
    num_removed_nodes = tree.descendants[node_idx].item()

    if delete_all_children:
        num_removed_nodes += 1

    indices_to_keep = torch.cat(
        (
            torch.arange(node_idx if delete_all_children else node_idx + 1),  # keep node_idx
            torch.arange(node_idx + num_removed_nodes if delete_all_children else node_idx + 1 + num_removed_nodes, len(tree))
        )
    )

    # select indices to keep from tokens
    if isinstance(tree.node_data, (torch.Tensor, np.ndarray)):
        node_data = tree.node_data[indices_to_keep]
    else:
        # handle lists
        node_data = [tree.node_data[index.item()] for index in indices_to_keep]

    # sequences in additional data are the same type as in node data
    if isinstance(tree.node_data, (torch.Tensor, np.ndarray)):
        additional_data = [data[indices_to_keep] for data in tree.additional_data]
    else:
        # handle lists
        additional_data = [
            [data[index.item()] for index in indices_to_keep] for data in tree.additional_data
        ]

    # and from parent and descendant tensors
    parents = tree.parents[indices_to_keep]
    descendants = tree.descendants[indices_to_keep]

    # explicitly set masked token
    if not delete_all_children:
        descendants[node_idx] = 0
        node_data[node_idx] = replacement

    # Adjust parents after new mask_pos
    parents[node_idx + 1:][parents[node_idx + 1:] > node_idx + 1] -= num_removed_nodes

    # go through each parent of node at mask_pos and adjust descendants
    for ancestor in tree.iter_ancestors(node_idx):
        descendants[ancestor] -= num_removed_nodes

    return tensortree.tree(node_data=node_data, parents=parents, descendants=descendants, additional_data=additional_data)


def delete_siblings(tree: TensorTree, node_indices: Union[int, torch.Tensor], replacement: Optional[Any] = None) -> TensorTree:
    """
    Returns a new tree with branches at node_indices deleted or replaced with a single node without children.
    The node indices must be direct siblings!

    """
    if replacement is not None:
        assert isinstance(replacement, (torch.Tensor, int, type(tree.node_data[node_indices[0]]))), "Replacement token needs to be tensor type"

    if not isinstance(node_indices, torch.Tensor):
        node_indices = torch.tensor(node_indices, dtype=torch.long)

    # maybe check if node_indices are really siblings, otherwise it will produce garbarge

    # which nodes to keep
    insert_replacement = replacement is not None

    start = node_indices[0]
    end = tree.next_node_not_in_branch(node_indices[-1])
    if end is not None:
        indices_to_keep = torch.cat(
            (
                torch.arange(start if not insert_replacement else start + 1),  # keep node_idx
                torch.arange(end, len(tree))
            )
        )
    else:
        indices_to_keep = torch.arange(start if not insert_replacement else start + 1)  # keep node_idx

    # select indices to keep from tokens
    if isinstance(tree.node_data, (torch.Tensor, np.ndarray)):
        node_data = tree.node_data[indices_to_keep]
    else:
        # handle lists
        node_data = [tree.node_data[index.item()] for index in indices_to_keep]

    # sequences in additional data are the same type as in node data
    if isinstance(tree.node_data, (torch.Tensor, np.ndarray)):
        additional_data = [data[indices_to_keep] for data in tree.additional_data]
    else:
        # handle lists
        additional_data = [
            [data[index.item()] for index in indices_to_keep] for data in tree.additional_data
        ]

    # and from parent and descendant tensors
    parents = tree.parents[indices_to_keep]
    descendants = tree.descendants[indices_to_keep]

    num_removed_nodes = len(tree) - len(indices_to_keep)

     # Adjust parents after new mask_pos
    parents[start:][parents[start:] > start] -= num_removed_nodes

    # go through each parent of node at mask_pos and adjust descendants
    for ancestor in tree.iter_ancestors(start):
        descendants[ancestor] -= num_removed_nodes

    # explicitly set masked token
    if insert_replacement:
        descendants[start] = 0
        node_data[start] = replacement

    return tensortree.tree(node_data=node_data, parents=parents, descendants=descendants, additional_data=additional_data)


def delete_children(
        tree: TensorTree, node_idx: int, replacement_token: Any = 999,
) -> TensorTree:
    """
    Parents is expected to be a tensor, which can be used to index the tensors (remove any padding or
    incrementations due to padding before !). Roots parent idx should be -1.

    Does the following, given this tree and node_idx=2, mask_token=99:

    0 MethodDeclaration
    ├── 1 parameters
    │   ├── 2 FormalParameter
    │   │   ├── 3 type
    │   │   │   └── 4 ReferenceType
    │   │   │       └── 5 name
    │   │   │           └── 6 Bitmap
    │   │   └── 7 name
    │   │       └── 8 bmp
    │   └── 9 FormalParameter
    │       ├── 10 type
    │       │   └── 11 ReferenceType
    │       │       └── 12 name
    │       │           └── 13 File
    │       └── 14 name
    └── 15 body

    Will return tensors for the following tree:

    0 MethodDeclaration (0)
    ├── 1 parameters (1)
    │   ├── 2 FormalParameter (2)
    │   │   └── 3 <MASK> (3)
    │   └── 9 FormalParameter (4)
    │       ├── 10 type (5)
    │       │   └── 11 ReferenceType (6)
    │       │       └── 12 name (7)
    │       │           └── 13 File (8)
    │       └── 14 name (9)
    └── 15 body (10)

    The original tensors have been:
    tokens:        [  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15]
    parents:       [ -1,  0,  1,  2,  3,  4,  5,  2,  7,  1,  9, 10, 11, 12,  9,  0]
    #descendants:  [ 14, 13,  6,  3,  2,  1,  0,  1,  0,  5,  3,  2,  1,  0,  0,  0]
    #children:     [  2,  2,  2,  1,  1,  1,  0,  1,  0,  2,  1,  1,  1,  0,  0,  0]

    This method will return the following tensors:
    words:        [ 0, 1, 2, 99, 9, 10, 11, 12, 13, 14, 15]
    parents:      [-1, 0, 1,  2, 1,  4,  5,  6,  7,  4,  0]
    #descendants: [10, 8, 1,  0, 5,  3,  2,  1,  0,  0,  0]
    #children:    [ 2, 2, 1,  0, 2,  1,  1,  1,  0,  0,  0]

    """
    if replacement_token is not None:
        assert isinstance(replacement_token,
                          (torch.Tensor, int, type(tree.node_data[node_idx]))), "Replacement token needs to be tensor type"

    pos_mask = node_idx + 1
    num_descendants_of_masked_node = tree.descendants[node_idx]
    num_removed_nodes = num_descendants_of_masked_node - 1

    assert num_descendants_of_masked_node > 0, "Node to mask children must have at least 1 child!"

    indices_to_keep = torch.cat(
        [
            torch.arange(node_idx + 2),  # keep node_idx and 1 child (which we will mask)
            torch.arange(node_idx + num_descendants_of_masked_node + 1, tree.descendants.size(0))
        ]
    )
    parents = tree.parents[indices_to_keep]
    descendants = tree.descendants[indices_to_keep]

    if is_tensor_type(tree.node_data):
        node_data = tree.node_data[indices_to_keep]
    else:
        # handle other sequence types, such as list of strings
        node_data = [tree.node_data[index.item()] for index in indices_to_keep]

    # sequences in additional data are the same type as in node data
    if isinstance(tree.node_data, (torch.Tensor, np.ndarray)):
        additional_data = [data[indices_to_keep] for data in tree.additional_data]
    else:
        # handle lists
        additional_data = [
            [data[index.item()] for index in indices_to_keep] for data in tree.additional_data
        ]

    node_data[pos_mask] = replacement_token
    descendants[node_idx] = 1
    descendants[pos_mask] = 0

    # Adjust parents after new mask_pos
    parents[node_idx + 1:][parents[node_idx + 1:] > node_idx + 1] -= num_removed_nodes

    # go through each parent of node at mask_pos and adjust descendants
    for ancestor in tree.iter_ancestors(node_idx):
        descendants[ancestor] -= num_removed_nodes

    return tensortree.tree(node_data=node_data, parents=parents, descendants=descendants, additional_data=additional_data)


def swap(tree: TensorTree, node_1: Union[int, torch.Tensor], node_2: Union[int, torch.Tensor]):
    """
    Does the following given this tree and 1, 8 for the swap positions:

    before swap:

    0 MethodDeclaration
    ├── 1 BasicType
    │   └── 2 int
    ├── 3 name
    │   └── 4 roundToDimensions
    ├── 5 FormalParameter
    │   ├── 6 BasicType
    │   │   └── 7 int
    │   └── 8 value
    ├── 9 FormalParameter
    │   ├── 10 BasicType
    │   │   └── 24 int
    │   └── 11 nearestValue
    ...

    after the swap:

    0 MethodDeclaration
    ├── 1 value
    ├── 2 name
    │   └── 3 roundToDimensions
    ├── 4 FormalParameter
    │   ├── 5 BasicType
    │   │   └── 6 int
    │   └── 7 BasicType
    │       └── 8 int
    ├── 9 FormalParameter
    │   ├── 10 BasicType
    │   │   └── 11 int
    │   └── 12 nearestValue
    ...
    The original tensors have been:
        tokens:        [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, ...]
        parents:       [-1,  0,  1,  0,  3,  0,  5,  6,  5,  0,  9, 10,  9, ...]
        #descendants:  [27,  1,  0,  1,  0,  3,  1,  0,  0,  3,  1,  0,  0, ...]
        #children:     [ 5,  1,  0,  1,  0,  2,  1,  0,  0,  2,  1,  0,  0, ...]

    This method will return the following tensors:
        words:        [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, ...]
        parents:      [-1,  0,  1,  0,  3,  0,  5,  6,  5,  0,  9, 10,  9, ...]
        #descendants: [27,  1,  0,  1,  0,  3,  1,  0,  0,  3,  1,  0,  0, ...]
        #children:    [ 5,  1,  0,  1,  0,  2,  1,  0,  0,  2,  1,  0,  0, ...]


    Here a sketch to understand the meaning of the variables defined further down:

                        tree_start_diff
                     <---------------------------->
    +----------------+--------+-------------------+------------------------+---------+
    |trees be4 tree 1| tree 1 |t. between swap t. |        tree 2          |t. after | <--- Tensor before swap
    +----------------+--------+-------------------+------------------------+---------+
                     ^        ^                   ^                        ^
                     |        |                   |                        |
                     +        +                   +                        +
                  node_1    node_1_end          node_2               node_2_end
                     +  node_2_swp_end       node_1_swp_start              +
                     |        +                   +                        |
                     |        |                   |                        |
                     v        +---------------v   +---------------v        v
    +----------------+------------------------+-------------------+--------+---------+
    |trees be4 tree 1|        tree 2          |t. between swap t. | tree 1 |t. after | <--- Tensor after swap
    +----------------+------------------------+-------------------+--------+---------+
                                                  <--------------->
                                                  tree_length_diff

    """

    if node_1 == node_2:  # nothing to do
        return tree.detach()
    elif node_1 > node_2:
        node_1, node_2 = node_2, node_1

    # + 1's are for easier slicing
    size = len(tree)
    tree_length_diff = tree.get_number_of_descendants(node_1) - tree.get_number_of_descendants(node_2)
    tree_start_diff = node_2 - node_1

    node_1_end = tree.next_node_not_in_branch(node_1)
    node_2_end = tree.next_node_not_in_branch(node_2)

    print("node_1_end, node_2_end:", node_1_end, node_2_end)
    # node_1_end = node_1 + tree.get_number_of_descendants(node_1)
    # node_2_end = node_2 + tree.get_number_of_descendants(node_2) + 1

    node_1_swp_start = node_2 - tree_length_diff
    node_2_swp_end = node_1 + tree.get_number_of_descendants(node_2) + 1

    # the start of the first swapped tree and the end of
    # the second swapped tree do not change
    # node_1_end += 1  # easier slicing

    # Check if the second tree is a sub-tree of the first one
    if ((node_1_end - 1) if node_1_end is not None else len(tree) + 1) >= node_2:
        raise ValueError(f"Node {node_2} is a subtree of node {node_1}")

    def swap_node_data() -> torch.Tensor:
        original_node_data = tree.node_data
        swapped_node_data = original_node_data.clone()  # create a copy

        swapped_node_data[node_1_swp_start:node_2_end] = original_node_data[node_1:node_1_end]  # copy subtree 1
        swapped_node_data[node_2_swp_end:node_1_swp_start] = original_node_data[node_1_end:node_2]  # move nodes between trees
        swapped_node_data[node_1:node_2_swp_end] = original_node_data[node_2:node_2_end]  # copy subtree 2
        return swapped_node_data

    def swap_additional_data() -> List[torch.Tensor]:
        swapped_additional_data = []

        for data in tree.additional_data:
            swapped_data = data.clone()  # create a copy

            swapped_data[node_1_swp_start:node_2_end] = data[node_1:node_1_end]  # copy subtree 1
            swapped_data[node_2_swp_end:node_1_swp_start] = data[node_1_end:node_2]  # move nodes between trees
            swapped_data[node_1:node_2_swp_end] = data[node_2:node_2_end]  # copy subtree 2

            swapped_additional_data.append(swapped_data)

        return swapped_additional_data

    def swap_parents() -> torch.Tensor:
        original_parents = tree.parents
        swapped_parents = original_parents.clone()

        swapped_parents[node_1_swp_start + 1:node_2_end] = original_parents[node_1 + 1:node_1_end]  # copy subtree 1
        swapped_parents[node_1_swp_start + 1:node_2_end] += tree_start_diff - tree_length_diff  # adjust

        # since the trees should swap we have to skip the first parents
        beginning_2 = node_2 + 1
        beginning_2_swp = node_1_swp_start + 1
        # all Nodes which parents are between the swap-trees
        mask = original_parents[node_1_end:beginning_2] > node_1

        # parent points to node be4 swap (no adjustments) or after swap (needs to be adjusted)
        swapped_parents[node_2_swp_end:beginning_2_swp][mask] = \
            original_parents[node_1_end:beginning_2][mask] - tree_length_diff
        swapped_parents[node_2_swp_end:beginning_2_swp][~mask] = \
            original_parents[node_1_end:beginning_2][~mask]

        # copy tree 2
        swapped_parents[node_1 + 1:node_2_swp_end] = \
            original_parents[node_2 + 1:node_2_end] - tree_start_diff

        # all Nodes after the swap which parents are between the swap-trees
        mask1 = (node_1_end - 1) < original_parents[node_2_end:size]
        mask2 = original_parents[node_2_end:size] < node_2
        swapped_parents[node_2_end:size][mask1 & mask2] -= tree_length_diff

        return swapped_parents

    def swap_descendants(swapped_parents) -> torch.Tensor:
        original_descendants = tree.descendants
        swapped_descendants = original_descendants.clone()

        # copy tree 1
        swapped_descendants[node_1_swp_start:node_2_end] = original_descendants[node_1:node_1_end]

        # Nodes between swap-trees
        swapped_descendants[node_2_swp_end:node_1_swp_start] = original_descendants[node_1_end:node_2]

        # copy tree 2
        swapped_descendants[node_1:node_2_swp_end] = original_descendants[node_2:node_2_end]

        # update the descendant count of the ancestors (note the ancestors did not change, so we can iter the
        # original tree)
        for ancestor in tree.iter_ancestors(node_1):
            swapped_descendants[ancestor] -= tree_length_diff

        for ancestor in tree.iter_ancestors(node_2):
            swapped_descendants[ancestor] += tree_length_diff

        return swapped_descendants

    node_data = swap_node_data()
    parents = swap_parents()
    descendants = swap_descendants(parents)
    additional_data = swap_additional_data()

    return tensortree.tree(
        parents=parents,
        descendants=descendants,
        node_data=node_data,
        additional_data=additional_data
    )


def insert_child(
        tree: TensorTree, parent_idx: int, child_node_data: Union[Any, TensorTree],
        right_sibling_idx: Optional[int] = None, child_additional_data: Optional[List[Union[List, torch.Tensor]]] = None,
) -> TensorTree:
    assert isinstance(
        child_node_data,
        (torch.Tensor, int, type(tree.node_data[parent_idx]), TensorTree)
    ), f"Replacement token needs to be tensor type"

    if child_additional_data is None:
        child_additional_data = []

    if right_sibling_idx is None:
        # get node after this branch
        idx_to_insert = tree.next_node_not_in_branch(parent_idx)
        if idx_to_insert is None:
            idx_to_insert = len(tree)
    else:
        idx_to_insert = right_sibling_idx
        if parent_idx != tree.get_parent(right_sibling_idx):
            raise IndexError(f"node at right_sibling_idx needs to have parent_idx as parent and not {tree.get_parent(right_sibling_idx)}")

    if isinstance(child_node_data, TensorTree):
        parents_to_insert = child_node_data.parents.clone()
        parents_to_insert[0] = parent_idx
        parents_to_insert[1:] += idx_to_insert

        descendants_to_insert = child_node_data.descendants
        node_data_to_insert = child_node_data.node_data
        additional_data_to_insert = child_node_data.additional_data
    else:
        parents_to_insert = tree.parents.new_tensor([parent_idx])
        descendants_to_insert = tree.descendants.new_tensor([0])

        if isinstance(tree.node_data, torch.Tensor):
            node_data_to_insert = tree.node_data.new_tensor([child_node_data])
            additional_data_to_insert = [orig_data.new_tensor([new_data]) for new_data, orig_data in zip(child_additional_data, tree.additional_data)]
        else:
            # handle lists
            node_data_to_insert = [child_node_data]
            additional_data_to_insert = [new_data for new_data in child_additional_data]

    if len(additional_data_to_insert) != len(tree.additional_data):
        raise ValueError(f"Need same amount of additional data. Tree: {len(tree)}, New node: {len(child_additional_data)}")

    # node data
    if isinstance(tree.node_data, torch.Tensor):
        node_data = torch.cat([
            tree.node_data[:idx_to_insert],
            node_data_to_insert,
            tree.node_data[idx_to_insert:],
        ])
        additional_data = [
            torch.cat([orig_data[:idx_to_insert], new_data, orig_data[idx_to_insert:],])
            for orig_data, new_data in zip(tree.additional_data, additional_data_to_insert)
        ]

    elif isinstance(tree.node_data, list):
        node_data = tree.node_data[:idx_to_insert] + node_data_to_insert + tree.node_data[idx_to_insert:]
        additional_data = [
            orig_data[:idx_to_insert] + new_data + orig_data[idx_to_insert:]
            for orig_data, new_data in zip(tree.additional_data, additional_data_to_insert)
        ]
    else:
        raise ValueError("Unknown type of node_data.")

    num_nodes_added = parents_to_insert.shape[0]

    # parents
    parents_after_insert = tree.parents[idx_to_insert:].clone()
    parents_after_insert[parents_after_insert >= idx_to_insert] += num_nodes_added

    parents = torch.cat([
        tree.parents[:idx_to_insert],
        parents_to_insert,
        parents_after_insert,
    ])

    descendants = torch.cat([
        tree.descendants[:idx_to_insert],
        descendants_to_insert,
        tree.descendants[idx_to_insert:],
    ])

    # go through each parent of node at mask_pos and adjust descendants
    descendants[parent_idx] += num_nodes_added
    for ancestor in tree.iter_ancestors(parent_idx):
        descendants[ancestor] += num_nodes_added

    return tensortree.tree(node_data=node_data, parents=parents, descendants=descendants, additional_data=additional_data)


def delete_nodes(tree: TensorTree, node_indices: Union[int, torch.Tensor], replacements: Optional[Any] = None, replacement_mask: torch.BoolTensor = None) -> TensorTree:
    """
    Returns a new tree with branches at node_indices deleted or replaced with a single node without children.
    The node indices must be in distinct branches!

    """

    if replacements is not None:
        assert isinstance(replacements, (torch.Tensor, list, type(tree.node_data[node_indices[0]]))), "Replacement token needs to be tensor type"
        if replacement_mask is None:
            assert len(replacements) == len(node_indices)

    if not isinstance(node_indices, torch.Tensor):
        node_indices = torch.tensor(node_indices, dtype=torch.long)

    if replacements is not None and is_tensor_type(tree.node_data) and not is_tensor_type(replacements):
        replacements = torch.tensor(replacements)

    if replacement_mask is not None:
        if replacements is None:
            raise ValueError("Replacements must be set when using replacements_mask")
        if len(replacement_mask) != len(node_indices):
            raise ValueError(f"Replacement mask must have same shape as node_indices.")
        assert replacement_mask[0], "first should be true"
    elif replacements is not None:
        replacement_mask = [True] * len(replacements)
    else:
        replacement_mask = []

    replacement_mask = torch.tensor(replacement_mask, dtype=torch.bool)

    # which nodes to keep
    insert_replacement = replacements is not None

    ####################
    start = 0
    end = node_indices[0]

    if insert_replacement:
        end = end + 1

    branch_positions = [
        torch.arange(start, end),
    ]

    for i in range(len(node_indices[:-1])):
        n_idx = node_indices[i]
        next_n_idx = node_indices[i+1]

        start = tree.next_node_not_in_branch(n_idx)
        end = next_n_idx

        if insert_replacement and replacement_mask[i+1]:
            end = end + 1
        branch_positions.append(
            torch.arange(start, end)
        )

    start = tree.next_node_not_in_branch(node_indices[-1])
    if start is not None:
        branch_positions.append(
            torch.arange(start, len(tree))
        )

    indices_to_keep = torch.cat(branch_positions)

    # select indices to keep from tokens
    if isinstance(tree.node_data, (torch.Tensor, np.ndarray)):
        node_data = tree.node_data[indices_to_keep]
    else:
        # handle lists
        node_data = [tree.node_data[index.item()] for index in indices_to_keep]

    # sequences in additional data are the same type as in node data
    if isinstance(tree.node_data, (torch.Tensor, np.ndarray)):
        additional_data = [data[indices_to_keep] for data in tree.additional_data]
    else:
        # handle lists
        additional_data = [
            [data[index.item()] for index in indices_to_keep] for data in tree.additional_data
        ]

    # and from parent and descendant tensors
    parents = tree.parents[indices_to_keep]
    descendants = tree.descendants[indices_to_keep]

    # go through each parent of node at mask_pos and adjust descendants
    num_removed = tree.descendants[node_indices] + 1

    if insert_replacement:
        num_removed[replacement_mask] -= 1

    kept_node_mask = torch.zeros_like(tree.descendants, dtype=torch.bool)
    kept_node_mask[indices_to_keep] = True
    num_removed_before = (~kept_node_mask).cumsum(-1)

    def to_new_idx(idx):
        return idx - num_removed_before[idx]

    new_node_indices = to_new_idx(node_indices)

    if insert_replacement:
        descendants[new_node_indices[replacement_mask]] = 0
        if isinstance(tree.node_data, (torch.Tensor, np.ndarray)):
            node_data[new_node_indices[replacement_mask]] = replacements
        else:
            num_replaced = 0
            for i in range(len(new_node_indices)):
                if replacement_mask[i]:
                    n_idx = new_node_indices[i]
                    replacement = replacements[num_replaced]
                    node_data[n_idx] = replacement
                    num_replaced += 1

    for i, n_idx in enumerate(node_indices):
        deleted_nodes = num_removed[i]
        new_n_idx = new_node_indices[i]

        start = new_n_idx
        if insert_replacement and replacement_mask[i]:
            start += 1

        parents[start:][parents[start:] > start] -= deleted_nodes

        if deleted_nodes > 0:
            for ancestor in tree.iter_ancestors(n_idx):
                ancestor = to_new_idx(ancestor)
                descendants[ancestor] -= deleted_nodes

    tree = tensortree.tree(node_data=node_data, descendants=descendants, parents=parents, additional_data=additional_data)
    return tree


def replace_children_of_nodes(tree: TensorTree, node_indices: Union[int, torch.Tensor], replacements: list[Any], replacement_additional_data: list[list[Any]] = None) -> TensorTree:
    """
    Returns a new tree with branches at node_indices deleted or replaced with a single node without children.
    The node indices must be in distinct branches!

    """
    if not node_indices:
        return tree

    assert isinstance(replacements, list), "Replacement token needs to be a listtype"
    assert isinstance(replacements[0], (list, type(tree.node_data[node_indices[0]]))), "replacements need to be of same type as node data"
    assert len(node_indices) == len(replacements)

    if not isinstance(node_indices, torch.Tensor):
        node_indices = torch.tensor(node_indices, dtype=torch.long)

    if is_tensor_type(tree.node_data) and not is_tensor_type(replacements[0]):
        for i in range(len(replacements)):
            replacements[i] = torch.tensor(replacements[i])

    ####################

    n = tree.node_data
    a = tree.additional_data
    d = tree.descendants
    p = tree.parents

    current = 0
    current_idx = node_indices[current]

    new_node_data = [n[:current_idx + 1]]
    new_descendants = [d[:current_idx + 1]]
    new_parents = [p[:current_idx + 1]]
    new_node_additional_data = [
        [add_data[:current_idx + 1]] for add_data in a
    ]

    num_nodes_removed = []
    to_new_idx = torch.arange(len(tree))

    sum_of_nodes = new_node_data[0].size(0)
    new_node_indices = []

    for i in range(len(node_indices)):
        last = i
        last_idx = node_indices[last]

        ###############
        # insert replacement
        ###############
        new_node_indices.append(sum_of_nodes - 1)

        node_data_replacement = replacements[i]
        new_node_data.append(node_data_replacement)
        sum_of_nodes += len(node_data_replacement)

        descendant_replacement = torch.zeros_like(node_data_replacement)  # fixme maybe replace with replacementtree.descendants
        new_descendants.append(descendant_replacement)

        parents_replacement = torch.zeros_like(node_data_replacement)  # fixme maybe replace with replacementtree.parents
        parents_replacement += last_idx
        new_parents.append(parents_replacement)

        num_descendants_of_last = tree.get_number_of_descendants(last_idx)
        num_removed_for_node = num_descendants_of_last - len(node_data_replacement)
        num_nodes_removed.append(num_removed_for_node)

        if replacement_additional_data is not None:
            additional_data_replacement = replacement_additional_data[i]
            for i_, add_data in enumerate(additional_data_replacement):
                new_node_additional_data[i_].append(add_data)

        ###############
        # then node
        ###############
        current = i+1
        start = tree.next_node_not_in_branch(last_idx)

        if start is not None:
            end = node_indices[current] + 1 if current < len(node_indices) else None

            node_data_added = n[start:end]
            new_node_data.append(node_data_added)
            new_descendants.append(d[start:end])
            new_parents.append(p[start:end])

            sum_of_nodes += len(node_data_added)

            if num_removed_for_node > 0:
                to_new_idx[(start - num_removed_for_node):start] = 99999 # something invalid
                to_new_idx[start:] -= num_removed_for_node
            else:
                to_new_idx[start:] -= num_removed_for_node

            for i_, add_data in enumerate(a):
                new_node_additional_data[i_].append(add_data[start:end])

    if is_tensor_type(tree.node_data):
        new_node_data = torch.cat(new_node_data)
        new_node_additional_data = [torch.cat(add_data) for add_data in new_node_additional_data]
    else:
        new_node_data = [x for chunk in new_node_data for x in chunk]
        new_node_additional_data = [[x for chunk in add_data for x in chunk] for add_data in new_node_additional_data]

    new_descendants = torch.cat(new_descendants)
    new_parents = torch.cat(new_parents)

    # adjust parents
    new_parents = to_new_idx[new_parents]
    new_parents[0] = -1
    # and decendants
    for node_idx, new_node_idx, num_removed_for_node, replacement in zip(node_indices, new_node_indices, num_nodes_removed, replacements):
        for ancestor in tree.iter_ancestors(node_idx):
            ancestor_new = to_new_idx[ancestor]
            new_descendants[ancestor_new] -= num_removed_for_node

        new_descendants[new_node_idx] = len(replacement)

    new_tree = tensortree.tree(node_data=new_node_data, descendants=new_descendants,
                               parents=new_parents, additional_data=new_node_additional_data)

    return new_tree


def cat(trees: List[tensortree.TensorTree], new_root_node_data: Any, new_root_additional_data: list[Any] = None):
    if new_root_additional_data is None:
        new_root_additional_data = []
    if len(new_root_additional_data) != len(trees[0].additional_data):
        raise ValueError("Additional data must have same shape")

    is_tensor_data = is_tensor_type(trees[0].node_data)

    # set elements of new root node
    if is_tensor_data:
        if not isinstance(new_root_node_data, torch.Tensor):
            new_root_node_data = torch.tensor([new_root_node_data])

        new_root_additional_data = [
            [torch.tensor(ad) if not isinstance(ad, torch.Tensor) else ad] for ad in new_root_additional_data
        ]
        for add_data in new_root_additional_data:
            if add_data and add_data[0].ndim == 0:
                add_data[0] = add_data[0].unsqueeze(0)

    node_data = [new_root_node_data]
    additional_data = new_root_additional_data

    descendants = [None]  # will be set later
    parents = [torch.tensor([-1])]

    sum_of_nodes = len(node_data)
    for tree in trees:
        descendants.append(tree.descendants)

        # shift parents
        p = tree.parents  # starts with -1
        p += sum_of_nodes
        p[0] = 0  # root
        sum_of_nodes += len(tree)
        parents.append(p)

        if is_tensor_data:
            node_data.append(tree.node_data)
            for i, add_data in enumerate(tree.additional_data):
                if add_data.ndim == 0:
                    add_data = add_data.unsqueeze(0)

                additional_data[i].append(add_data)
        else:
            node_data.extend(tree.node_data)
            for i, add_data in enumerate(tree.additional_data):
                additional_data[i].extend(add_data)

    # concatenate everything
    if is_tensor_data:
        node_data = torch.cat(node_data)
        additional_data = [torch.cat(add_data) for add_data in additional_data]

    parents = torch.cat(parents)

    num_descendants_of_root = len(node_data) - 1
    descendants[0] = torch.tensor([num_descendants_of_root])
    descendants = torch.cat(descendants)

    new_tree = tensortree.tree(
        node_data=node_data, descendants=descendants,
        parents=parents, additional_data=additional_data
    )

    return new_tree


def delete_nodes_and_return_leaves(
        tree: tensortree.TensorTree, indices_of_nodes: torch.Tensor, replacement_tokens: torch.Tensor
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Replaces node at indices with tokens and returns all leaves of the resulting tree..

    :param tree: A tree.
    :param indices_of_nodes: 1-d long tensor of node indices to replace (Shape: [S])
    :param replacement_tokens: 1-d tensor of replacement tokens of same shape as `indices_of_nodes`
    :return:
    """
    assert indices_of_nodes.size(0) == replacement_tokens.size(0), "indices of nodes and replacement_tokens should have same shape"
    assert (1 == indices_of_nodes.dim() == replacement_tokens.dim()), "indices of nodes and replacement_tokens should have same shape"

    leaf_mask = tree.leaves_mask()
    number_of_nonterminals_before = (~leaf_mask).cumsum(0) - 1

    # calc the postition of the indices in masked token array
    adj_indices_of_nodes = indices_of_nodes - number_of_nonterminals_before[indices_of_nodes]

    # end position in original node_data array (with nonterminals)
    indices_of_nodes_end = indices_of_nodes + tree.descendants[indices_of_nodes]

    # end position in masked token array
    adj_indices_of_nodes_end = indices_of_nodes_end - number_of_nonterminals_before[indices_of_nodes_end]

    indices_of_nodes = adj_indices_of_nodes
    indices_of_nodes_end = adj_indices_of_nodes_end

    # We then obtain slices for each part before a node to be removed.
    # relevant_slice, node1_part, relevant_slice, node2_part, ...
    # 2*S indices_or_sections argument for torch.tensor_split. E.g. indices_or_sections=[2, 3]
    # and dim=0 would result in the tensors input[:2], input[2:3], and input[3:].
    split_positions = torch.hstack((indices_of_nodes[:, None], indices_of_nodes_end[:, None]),).view(-1)
    relevant_slices = torch.tensor_split(tree.node_data[leaf_mask], split_positions)

    replaced_parts = relevant_slices[1::2]
    relevant_slices = relevant_slices[::2]

    assert len(replaced_parts) == replacement_tokens.size(0)
    # and concatenate the separate pieces
    res = torch.cat([x for pair in zip_longest(relevant_slices, replacement_tokens[:, None]) for x in pair if x is not None])
    return res, replaced_parts


# def _replace_nodes_and_return_leaves_list_comp(tree: tensortree.TensorTree, indices_of_nodes: torch.Tensor, replacement_tokens: torch.Tensor):
#     """
#     Replaces list of nodes with a new leaf token and returns all leaves of the resulting tree..
#
#     :param tree: A tree
#     :param indices_of_nodes: 1-d long tensor of node indices to replace (Shape: [S])
#     :param replacement_tokens: 1-d tensor of replacement tokens of same shape as `indices_of_nodes`
#     :return:
#     """
#     assert indices_of_nodes.size(0) == replacement_tokens.size(0), "indices of nodes and replacement_tokens should have same shape"
#     assert (1 == indices_of_nodes.dim() == replacement_tokens.dim()), "indices of nodes and replacement_tokens should have same shape"
#     # tree.pprint()
#     # last token of the node's subtrees
#     indices_of_nodes_end = indices_of_nodes + tree.descendants[indices_of_nodes]  # I
#
#     # We then obtain slices for each part before a node to be removed.
#     # relevant_slice, node1_part, relevant_slice, node2_part, ...
#     split_positions = torch.hstack(
#         (
#             indices_of_nodes[:, None],
#             (indices_of_nodes_end + 1)[:, None]
#         ),
#     ).view(-1)  # 2*S indices_or_sections argument for torch.tensor_split. E.g. indices_or_sections=[2, 3]
#     # and dim=0 would result in the tensors input[:2], input[2:3], and input[3:].
#     print(split_positions)
#     print(tree.leaves_mask().byte())
#     print(tree.leaves_mask().byte())
#     print(tree.node_data)
#     relevant_slices = torch.tensor_split(tree.node_data, split_positions)
#     relevant_slices_leaf_masks = torch.tensor_split(tree.leaves_mask(), split_positions)
#     # Then mask each slice
#     mask_leaves_before = (
#         t[mask] for t, mask in islice(zip(relevant_slices, relevant_slices_leaf_masks), None, None, 2)
#     )
#
#     # and concatenate the separate pieces
#     res = torch.cat([x for pair in zip_longest(mask_leaves_before, replacement_tokens[:, None]) for x in pair if x is not None])
#     return res
