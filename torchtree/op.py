import collections
from typing import Union, Any, Optional, Generator, List, Dict, Sequence

import numpy as np
import torch


from torchtree import new_tree, TensorTree


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


def node_incidence_matrix(
        descendants: torch.Tensor, pad_idx: int = -1, pad_mask: Optional[torch.Tensor] = None
) -> torch.BoolTensor:
    """
    Computes the node incidence matrix between nodes based on an
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


def levels(node_incidences: torch.Tensor) -> torch.Tensor:
    """ Computes the level/depth of each node based on a binary node incidence matrix.

    Root has level 1.

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

    tensor([1, 2, 2, 3, 3, 4], dtype=torch.int64)
    """
    return node_incidences.sum(dim=-1)


def adjacency_matrix(parents: torch.Tensor, directed: bool = False, direction_up: bool = False, pad_mask=None):

    batched = parents.ndim == 2
    if not batched:
        parents = parents[None, :]

    B, S = parents.shape

    if batched and pad_mask is None:
        pad_mask = parents < -1

    parents[:, 0], root_vals = 0, parents[:, 0]  # only if right padded

    print("root_vals", root_vals)
    print("->", parents)

    a = torch.arange(S, device=parents.device)
    b = torch.arange(B, device=parents.device)
    adj = parents.new_zeros(B, S, S, dtype=torch.bool)

    if directed and direction_up:
        adj[a, parents] = True
    elif directed and not direction_up:
        adj[parents] = True
    else:
        adj[a, parents] = True
        adj[parents, a] = True

    parents[:, 0] = root_vals

    return adj


def adjacency_matrix_unbatched(parents: torch.Tensor, directed: bool = False, direction_up: bool = False):
    parents[0], root_val = 0, parents[0]
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
    return adj


def ancestral_matrix(node_incidences: torch.Tensor) -> torch.Tensor:
    """
    Computes the length of the common prefix between two nodes (which
    is the same as the level of the least common ancestor).

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

    return res


def movements(node_incidences: torch.Tensor, pad_idx: int = -1, pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
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

    node_levels = level(node_incidences)
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


def least_common_ancestors(num_descendants: torch.Tensor) -> torch.Tensor:
    """
    Computes least common ancestors for a 1D tensor of num descendants
    :param num_descendants:  shape [S]
    :return:
    """
    # FIXME/TODO  make it work with padded batches, currently only works with 1D
    node_incidences = node_incidence_matrix(num_descendants, pad_idx=-1,)
    node_incidences = node_incidences.byte()
    m = node_incidences.unsqueeze(1) + node_incidences
    res = torch.argmax(m, dim=-1)

    return res


def parents_from_descendants(descendants: Sequence[int]) -> torch.Tensor:
    """ not very performant, but it works."""

    stack_idx = [0]
    stack_open_descendants = [descendants[0]]

    parents = [-1]

    for original_idx, num_descendants in enumerate(descendants[1:], start=1):
        parents.append(stack_idx[-1])

        stack_idx.append(original_idx)
        stack_open_descendants.append(num_descendants + 1)

        stack_open_descendants = [d - 1 for d in stack_open_descendants if (d - 1) > 0]
        stack_idx = stack_idx[:len(stack_open_descendants)]

    return descendants.new_tensor(
        parents
    ) if isinstance(descendants, torch.Tensor) else torch.tensor(parents, dtype=torch.int)


def descendants_from_parents(parents: Sequence[int]) -> torch.Tensor:
    descendants = parents.new_zeros(
        parents.size(-1)
    ) if isinstance(parents, torch.Tensor) else torch.zeros(len(parents), dtype=torch.int)

    active = torch.full_like(descendants, fill_value=False, dtype=torch.bool)  # bool tensor with all false

    for node_idx, parent_idx in enumerate(parents):
        active[(parent_idx + 1):] = False  # deactivate closed branch
        descendants[active] += 1  # increment descendants on all active nodes
        active[node_idx] = True  # set current node as active

    return descendants


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
        assert isinstance(replacement, (torch.Tensor, int, type(tree.data.node_data[node_idx]))), "Replacement token needs to be tensor type"

    delete_all_children = replacement is None

    # which nodes to keep
    num_removed_nodes = tree.data.descendants[node_idx].item()

    if delete_all_children:
        num_removed_nodes += 1

    indices_to_keep = torch.cat(
        (
            torch.arange(node_idx if delete_all_children else node_idx + 1),  # keep node_idx
            torch.arange(node_idx + num_removed_nodes if delete_all_children else node_idx + 1 + num_removed_nodes, len(tree))
        )
    )

    # select indices to keep from tokens
    if isinstance(tree.data.node_data, (torch.Tensor, np.ndarray)):
        node_data = tree.data.node_data[indices_to_keep]
    else:
        # handle lists
        node_data = [tree.data.node_data[index.item()] for index in indices_to_keep]

    # and from parent and descendant tensors
    parents = tree.data.parents[indices_to_keep]
    descendants = tree.data.descendants[indices_to_keep]

    # explicitly set masked token
    if not delete_all_children:
        descendants[node_idx] = 0
        node_data[node_idx] = replacement

    # Adjust parents after new mask_pos
    parents[node_idx + 1:][parents[node_idx + 1:] > node_idx + 1] -= num_removed_nodes

    # go through each parent of node at mask_pos and adjust descendants
    for ancestor in tree.iter_ancestors(node_idx):
        descendants[ancestor] -= num_removed_nodes

    return new_tree(node_data=node_data, parents=parents, descendants=descendants)



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
    parents = tree.data.parents[indices_to_keep]
    descendants = tree.data.descendants[indices_to_keep]

    if is_tensor_type(tree.data.node_data):
        node_data = tree.data.node_data[indices_to_keep]
    else:
        # handle other sequence types, such as list of strings
        node_data = [tree.data.node_data[index.item()] for index in indices_to_keep]

    node_data[pos_mask] = replacement_token
    descendants[node_idx] = 1
    descendants[pos_mask] = 0

    # Adjust parents after new mask_pos
    parents[node_idx + 1:][parents[node_idx + 1:] > node_idx + 1] -= num_removed_nodes

    # go through each parent of node at mask_pos and adjust descendants
    for ancestor in tree.iter_ancestors(node_idx):
        descendants[ancestor] -= num_removed_nodes

    return new_tree(node_data=node_data, parents=parents, descendants=descendants)


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
        return
    elif node_1 > node_2:
        node_1, node_2 = node_2, node_1

    # + 1's are for easier slicing
    size = len(tree)
    tree_length_diff = tree.data.descendants[node_1] - tree.data.descendants[node_2]
    tree_start_diff = node_2 - node_1

    node_1_end = node_1 + tree.data.descendants[node_1]
    node_2_end = node_2 + tree.data.descendants[node_2] + 1
    node_2_swp_end = node_1 + tree.data.descendants[node_2] + 1
    node_1_swp_start = node_2 - tree_length_diff
    # the start of the first swapped tree and the end of
    # the second swapped tree do not change

    # Check if the second tree is a sub-tree of the first one
    assert node_1_end < node_2, "The second tree is a subtree of the first one"

    node_1_end += 1  # easier slicing

    def swap_tokens() -> torch.Tensor:
        tokens_swp = tree.data.node_data.clone()
        tokens_swp[node_1_swp_start:node_2_end] = tree.data.node_data[node_1:node_1_end]  # copy subtree 1
        tokens_swp[node_2_swp_end:node_1_swp_start] = tree.data.node_data[node_1_end:node_2]  # move nodes between trees
        tokens_swp[node_1:node_2_swp_end] = tree.data.node_data[node_2:node_2_end]  # copy subtree 2
        return tokens_swp

    def swap_parents() -> torch.Tensor:
        parents_swp = tree.data.parents.clone()

        parents_swp[node_1_swp_start + 1:node_2_end] = tree.data.parents[node_1 + 1:node_1_end]  # copy subtree 1
        parents_swp[node_1_swp_start + 1:node_2_end] += tree_start_diff - tree_length_diff  # adjust

        # since the trees should swap we have to skip the first parents
        beginning_2 = node_2 + 1
        beginning_2_swp = node_1_swp_start + 1
        # all Nodes which parents are between the swap-trees
        mask = tree.data.parents[node_1_end:beginning_2] > node_1

        # parent points to node be4 swap (no adjustments) or after swap (needs to be adjusted)
        parents_swp[node_2_swp_end:beginning_2_swp][mask] = \
            tree.data.parents[node_1_end:beginning_2][mask] - tree_length_diff
        parents_swp[node_2_swp_end:beginning_2_swp][~mask] = \
            tree.data.parents[node_1_end:beginning_2][~mask]

        # copy tree 2
        parents_swp[node_1 + 1:node_2_swp_end] = \
            tree.data.parents[node_2 + 1:node_2_end] - tree_start_diff

        # all Nodes after the swap which parents are between the swap-trees
        mask1 = (node_1_end - 1) < tree.data.parents[node_2_end:size]
        mask2 = tree.data.parents[node_2_end:size] < node_2
        parents_swp[node_2_end:size][mask1 & mask2] -= tree_length_diff

        return parents_swp

    def swap_descendants(swapped_parents) -> torch.Tensor:
        result_descendants = tree.data.descendants.clone()

        # copy tree 1
        result_descendants[node_1_swp_start:node_2_end] = tree.data.descendants[node_1:node_1_end]

        # Nodes between swap-trees
        result_descendants[node_2_swp_end:node_1_swp_start] = tree.data.descendants[node_1_end:node_2]

        # copy tree 2
        result_descendants[node_1:node_2_swp_end] = tree.data.descendants[node_2:node_2_end]

        # update the descendant count of the ancestors
        current_parent = swapped_parents[node_1]
        while current_parent != -1:
            result_descendants[current_parent] -= tree_length_diff
            current_parent = swapped_parents[current_parent]

        current_parent = swapped_parents[node_1_swp_start]
        while current_parent != -1:
            result_descendants[current_parent] += tree_length_diff
            current_parent = swapped_parents[current_parent]

        return result_descendants

    node_data = swap_tokens()
    parents = swap_parents()
    descendants = swap_descendants(parents)

    return new_tree(
        parents=parents,
        descendants=descendants,
        node_data=node_data
    )


def insert_child(
        tree: TensorTree, parent_idx: int, child_node_data: Union[Any, TensorTree],
        right_sibling_idx: Optional[int] = None
) -> TensorTree:
    assert isinstance(
        child_node_data,
        (torch.Tensor, int, type(tree.data.node_data[parent_idx]), TensorTree)
    ), f"Replacement token needs to be tensor type"

    if right_sibling_idx is None:
        # get node after this branch
        idx_to_insert = tree.next_node_not_in_branch(parent_idx)
        if idx_to_insert is None:
            idx_to_insert = len(tree)
    else:
        idx_to_insert = right_sibling_idx
        if parent_idx != tree.parent(right_sibling_idx):
            raise IndexError(f"node at right_sibling_idx needs to have parent_idx as parent and not {tree.parent(right_sibling_idx)}")

    if isinstance(child_node_data, TensorTree):
        parents_to_insert = child_node_data.data.parents.clone()
        parents_to_insert[0] = parent_idx
        parents_to_insert[1:] += idx_to_insert

        descendants_to_insert = child_node_data.data.descendants
        node_data_to_insert = child_node_data.data.node_data

    else:
        parents_to_insert = tree.data.parents.new_tensor([parent_idx])
        descendants_to_insert = tree.data.descendants.new_tensor([0])

        if isinstance(tree.data.node_data, torch.Tensor):
            node_data_to_insert = tree.data.node_data.new_tensor([child_node_data])
        else:
            # handle lists
            node_data_to_insert = [child_node_data]

    # node data
    if isinstance(tree.data.node_data, torch.Tensor):
        node_data = torch.cat([
            tree.data.node_data[:idx_to_insert],
            node_data_to_insert,
            tree.data.node_data[idx_to_insert:],
        ])
    else:
        # handle lists
        node_data = tree.data.node_data[:idx_to_insert] + node_data_to_insert + tree.data.node_data[idx_to_insert:]

    num_nodes_added = parents_to_insert.shape[0]

    # parents
    parents_after_insert = tree.data.parents[idx_to_insert:].clone()
    parents_after_insert[parents_after_insert >= idx_to_insert] += num_nodes_added

    parents = torch.cat([
        tree.data.parents[:idx_to_insert],
        parents_to_insert,
        parents_after_insert,
    ])

    descendants = torch.cat([
        tree.data.descendants[:idx_to_insert],
        descendants_to_insert,
        tree.data.descendants[idx_to_insert:],
    ])

    # go through each parent of node at mask_pos and adjust descendants
    descendants[parent_idx] += num_nodes_added
    for ancestor in tree.iter_ancestors(parent_idx):
        descendants[ancestor] += num_nodes_added

    return new_tree(node_data=node_data, parents=parents, descendants=descendants)
