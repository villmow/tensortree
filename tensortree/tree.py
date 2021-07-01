from dataclasses import dataclass
from typing import Sequence, Any, Union, List, Optional, Tuple, Generator, Callable, Literal

import numpy as np
import torch

import tensortree
from tensortree.render import Style, ContRoundStyle, format_tree
from tensortree.utils import to_torch


# Define a type alias for the content of the node sequence
LabelType = Any
TensorType = Union[Sequence[int], np.ndarray, torch.Tensor]


@dataclass(frozen=True)
class TreeStorage:
    """
    Stores a tree with data.
    """

    # either parents or descendants may be None.
    # other sequence types will be converted to tensors.
    parents: TensorType = None
    descendants: Union[torch.Tensor, Sequence[int]] = None

    # some operations (swapping) only work when node_data is a torch.Tensor
    node_data: Union[torch.Tensor, Sequence[LabelType]] = None

    format: Literal['torch', 'numpy'] = 'torch'

    def __post_init__(self):
        if self.parents is None and self.descendants is None:
            raise ValueError("Either parents or descendants must be passed")

        if self.parents is None:  # compute parents from descendants
            descendants: torch.Tensor = to_torch(self.descendants).long()
            parents = tensortree.parents_from_descendants(self.descendants)

        elif self.descendants is None:  # compute descendants from parents
            parents: torch.Tensor = to_torch(self.parents).long()
            descendants = tensortree.descendants_from_parents(parents)

        else:  # convert everything to a tensor
            parents: torch.Tensor = to_torch(self.parents).long()
            descendants: torch.Tensor = to_torch(self.descendants).long()

        # node_data may be nothing, in that case simply enumerate the nodes
        if self.node_data is None:
            node_data = torch.arange(len(descendants)).to(descendants)
            # FIXME make node_data optional
        else:
            # node_data is a sequence of strings (tensor incompatible)
            try:
                node_data: torch.Tensor = to_torch(self.node_data).long()
            except (ValueError, TypeError, RuntimeError):
                node_data: List[LabelType] = list(self.node_data)

        if descendants.numel() != len(node_data) != parents.numel():
            raise ValueError(f"All arrays need to be of same length and not ({descendants.numel()}, {len(node_data)}, {parents.numel()}).")

        object.__setattr__(self, 'parents', parents)
        object.__setattr__(self, 'descendants', descendants)
        object.__setattr__(self, 'node_data', node_data)


def tree(
    parents: Optional[TensorType] = None,
    node_data: Optional[Sequence[LabelType]] = None,
    descendants: Optional[TensorType] = None,
):
    """ Constructor to build a tree. """

    return TensorTree.from_array(parents=parents, descendants=descendants, node_data=node_data)


class TensorTree:

    @classmethod
    def from_array(
            cls,
            parents: Optional[TensorType] = None, descendants: Optional[TensorType] = None,
            node_data: Optional[Sequence[LabelType]] = None
    ):
        """ Obtain a tree from arrays. A tree can be either defined by a parents or a descendants tensor.
        Additional nodes list can be passed if it contains a string, it will be used for rendering.
        """
        return cls(TreeStorage(parents, descendants, node_data))

    def __init__(self, data: TreeStorage, root_idx: int = 0):
        """
        Initialize with a pointer to tree storage.
        """
        self.data = data
        self.root_idx = root_idx

        if len(self.data.parents) > 0 and (self.data.parents[0] != -1):  # and self.data.parents[0] != 0):
            raise ValueError("Parents array seems to have wrong format.")

        self.__len = self.data.descendants.shape[-1]  # cache this

        # span in original array
        self.end = len(self)

        if self.is_subtree():
            self.__len = self.data.descendants[root_idx] + 1
            self.end = root_idx + len(self)

    def is_subtree(self) -> bool:
        return self.root_idx > 0

    def __len__(self):
        """ The number of nodes in this tree. """
        return self.__len

    def __getitem__(self, node_idx: Union[int, torch.Tensor, np.ndarray, Tuple[int, ...], None]):
        """ Will returns a view of the node at node_idx"""
        self._check_bounds(node_idx)

        if node_idx is None:
            node_idx = self.root_idx  # use root

        if node_idx == self.root_idx:
            return self

        return TensorTree(self.data, node_idx)

    def __str__(self):
        return self.pformat(max_nodes=4)

    def is_descendant_of(self, node_ancestor_idx: int, node_descendant_idx: int) -> bool:
        return node_ancestor_idx < node_descendant_idx < node_ancestor_idx + self.get_number_of_descendants(node_ancestor_idx)

    # helpers for individual nodes
    def get_node_data(self, node_idx: Union[int, torch.Tensor]) -> Any:
        """ Returns the label of a node. Node_idx refers to the index of the node in the original tree."""
        self._check_bounds(node_idx)
        return self.data.node_data[node_idx]

    def get_number_of_descendants(self, node_idx: Union[int, torch.Tensor]) -> torch.Tensor:
        """ Returns the amount of descendants of a node. Node_idx refers to the index of the node in the original tree."""
        self._check_bounds(node_idx)
        return self.data.descendants[node_idx]

    def get_parent(self, node_idx: Union[int, torch.Tensor]) -> Optional[int]:
        """ Returns the parent idx for this node or None if node_idx is root.

        :param node_idx: Node_idx refers to the index of the node in the original tree.
        :return:
        """
        self._check_bounds(node_idx)

        # return None for root
        if node_idx == self.root_idx:
            return

        return self.data.parents[node_idx]

    @property
    def descendants(self) -> torch.Tensor:
        """ Returns the relevant subset of descendants for this subtree."""
        if self.root_idx == 0:
            return self.data.descendants  # return full array
        else:
            return self.data.descendants[self.root_idx: self.end]  # return copy

    @property
    def node_data(self) -> Sequence[LabelType]:
        """ Returns the relevant subset of node_data for this subtree."""
        if self.root_idx == 0:
            return self.data.node_data
        else:
            return self.data.node_data[self.root_idx: self.end]

    @property
    def parents(self) -> torch.Tensor:
        """ Returns the relevant subset of parents for this subtree."""
        if self.root_idx == 0:
            return self.data.parents
        else:
            parents = self.data.parents[self.root_idx: self.end] - self.root_idx
            parents[0] = -1
            return parents

    def node_incidence_matrix(self):
        """ Returns the node incidence matrix for this subtree"""
        return tensortree.node_incidence_matrix(self.descendants)

    def detach(self):
        """ Returns a new tree rooted at self.root_idx """
        from copy import deepcopy
        return self.from_array(
            parents=self.parents.clone(),
            descendants=self.descendants.clone(),
            node_data=deepcopy(self.node_data)
        )

    def next_node_not_in_branch(self, node_idx: Union[int, torch.Tensor]) -> torch.Tensor:
        """
        Return the next node, that is not part of the branch of node at node_idx.

        Can be a sibling or any other node that follows after this branch.

        :param node_idx: Can be either an integer or a single value tensor
        :return:
        """
        next_node_idx = (node_idx + self.get_number_of_descendants(node_idx) + 1)
        return next_node_idx if next_node_idx < len(self) else None

    def step_out(self, node_idx: Union[int, torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Return the next node, that is not part of the subtree of node_idx's parent.

        Can be a sibling to nodes parent or any other node that follows.

        :param node_idx: Can be either an integer or a single value tensor
        :return:
        """
        nodes_parent = self.get_parent(node_idx)
        if nodes_parent is None:
            return

        return self.next_node_not_in_branch(nodes_parent)

    def node_idx_for_tree_position(self, tree_position: Tuple[int, ...]) -> Union[int, torch.Tensor]:
        if not isinstance(tree_position, tuple):
            raise ValueError("tree_position must be tuple")

        def get_nth_child(node_idx, n):
            for i, child_idx in enumerate(self.iter_children(node_idx)):
                if i == n:
                    return child_idx
            raise IndexError

        current_node_idx = self.root_idx  # root
        for child_number in tree_position:
            current_node_idx = get_nth_child(current_node_idx, child_number)

        return current_node_idx

    def depths(self):
        """ Depth of each subtree"""

        from tensortree import mask_layer
        depths = torch.zeros_like(self.descendants)

        for i, layer_mask in enumerate(mask_layer(self.node_incidence_matrix())):
            depths[layer_mask] = i

        return depths

    # leaves
    def is_leaf(self, node_idx: Union[int, torch.Tensor]) -> bool:
        return self.get_number_of_descendants(node_idx) == 0

    def leaves_mask(self) -> torch.BoolTensor:
        """ Returns a boolean mask for all leaf nodes in this tree """
        return (self.descendants == 0)

    def leaf_indices(self) -> torch.Tensor:
        return self.leaves_mask().nonzero().squeeze(-1)

    # children
    def iter_children(self, node_idx: Union[int, torch.Tensor]) -> Generator[torch.Tensor, None, None]:
        """
        Iters over the children of a node with a specific index in a tree.

        :param node_idx: Node to iterate over the children
        """
        self._check_bounds(node_idx)

        branch_end = self.next_node_not_in_branch(node_idx)  # end of subtree of node_idx

        if branch_end is None:
            branch_end = len(self)

        # next child is at the next position in the descendants array
        next_child = node_idx + 1

        # are we still in the subtree
        while next_child is not None and next_child < branch_end:
            yield next_child
            next_child = self.next_node_not_in_branch(next_child)

    def children_mask(self, node_idx: Union[int, torch.Tensor]) -> torch.BoolTensor:
        """
        Returns children indices of a node at a specific index in a tree.

        :param node_idx: Node to get the children
        """
        self._check_bounds(node_idx)

        return self.data.parents == node_idx

    def children(self, node_idx: Union[int, torch.Tensor]) -> torch.Tensor:
        """
        Returns children indices of a node at a specific index in a tree.

        :param node_idx: Node to get the children
        """
        return self.children_mask(node_idx).nonzero(False).squeeze(-1)

    # siblings
    def iter_siblings(self, node_idx: Union[int, torch.Tensor], include_left_siblings: bool = True, include_right_siblings: bool = True) -> Generator[torch.Tensor, None, None]:
        """ Node indices with the same parent. The node at node_idx is excluded.

        set include_left_siblings to False to only output siblings which are to the right of node at node_idx.
        set include_right_siblings to False  to only output siblings which are to the left of node at node_idx.
        """
        if not (include_left_siblings or include_right_siblings):
            raise ValueError("Left or right must be specified.")
        self._check_bounds(node_idx)

        parent_idx = self.get_parent(node_idx)

        # root
        if parent_idx is None:
            return

        reached_node = False
        for node in self.iter_children(parent_idx):
            if node == node_idx:
                reached_node = True
                continue

            if include_left_siblings and not reached_node:
                yield node

            if include_right_siblings and reached_node:
                yield node

    def right_sibling(self, node_idx: Union[int, torch.Tensor]) -> Optional[torch.Tensor]:
        next_node = self.next_node_not_in_branch(node_idx)
        if next_node is not None:
            if self.get_parent(next_node) == self.get_parent(node_idx):
                return next_node

    def siblings_mask(self, node_idx: Union[int, torch.Tensor], include_left_siblings: bool = True, include_right_siblings: bool = True) -> torch.BoolTensor:
        """ Node indices with the same parent. The node at node_idx is excluded.

        set include_left_siblings to False to only output siblings which are to the right of node at node_idx.
        set right to False  to only output siblings which are to the left of node at node_idx.
        """
        if not (include_left_siblings or include_right_siblings):
            raise ValueError("Left or right must be specified.")
        self._check_bounds(node_idx)

        parent_idx = self.get_parent(node_idx)

        # root
        if parent_idx is None:
            return self.data.descendants.new_zeros(len(self)).bool()

        all_siblings = self.children_mask(parent_idx)

        assert all_siblings[node_idx], "this should be true"
        # exlude node at node_idx
        all_siblings[node_idx] = False

        # exclude everything before node_idx
        if not include_left_siblings:
            all_siblings[:node_idx] = False

        # exclude everything after node_idx
        if not include_right_siblings:
            all_siblings[node_idx + 1:] = False

        return all_siblings

    def siblings(self, node_idx: Union[int, torch.Tensor], check_left: bool = True, check_right: bool = True) -> torch.Tensor:
        return self.siblings_mask(node_idx, check_left, check_right).nonzero().squeeze(-1)

    def has_sibling(self, node_idx: Union[int, torch.Tensor], check_left: bool = True, check_right: bool = True) -> bool:
        """
        Is there a sibling to this node?

        :param node_idx:
        :param check_left: look left for siblings (set to False to only check for right siblings)
        :param check_right: look right for siblings (set to False to only check for left siblings)
        :return:
        """
        try:
            self._check_bounds(node_idx)
        except IndexError:
            pass

        node_parent = self.get_parent(node_idx)
        if node_parent is None:
            return False

        result = False
        if check_right:
            next_node = self.next_node_not_in_branch(node_idx)
            if next_node is not None:
                next_nodes_parent = self.get_parent(next_node)
                has_right_sibling = (next_nodes_parent == node_parent)
                result = result or has_right_sibling
        if check_left and not result:
            # otherwise something should be between parent and node
            has_left_sibling = (node_parent + 1) != node_idx
            result = result or has_left_sibling

        return result

    # ancestors
    def iter_ancestors(self, node_idx: Union[int, torch.Tensor]) -> Generator[torch.Tensor, None, None]:
        node_parent = self.get_parent(node_idx)
        while node_parent is not None:
            yield node_parent
            node_parent = self.get_parent(node_parent)

    # pretty printing
    def pformat(
            self, max_nodes: Optional[int] = None, node_renderer: Callable[[Any], str] = str,
            style: Union[Style] = ContRoundStyle,
    ) -> str:
        """
        Pretty prints a tree up to `max_nodes`. Define a node_renderer for custom node types (e.g. Dictionaries).
        :param max_nodes: Render up to this amount of nodes.
        :param node_renderer: A function that outputs a string.
        :param style: Style the tree.
        :return:
        """

        return format_tree(tree=self, max_nodes=max_nodes, node_renderer=node_renderer, style=style)

    def pprint(
            self, max_nodes: Optional[int] = None, node_renderer: Callable[[Any], str] = str,
            style: Union[Style] = ContRoundStyle,
    ):
        """ See pformat for description of arguments."""
        print("TensorTree():\n", self.pformat(max_nodes, node_renderer, style))

    def _check_bounds(self, node_idx: Union[int, torch.Tensor]) -> None:
        """ Checks whether node_idx is in the bounds of this tree and raises exception if not """
        if node_idx < self.root_idx or node_idx >= self.end:
            raise IndexError(
                f"Index {node_idx} is out of bounds for this"
                f" {'sub' if self.is_subtree() else ''}tree starting at index"
                f" {self.root_idx} with {len(self)} nodes."
            )

    # functions below return modify and return a new tree
    def delete_node(self, node_idx: int, replacement_token: Optional[Any] = None):
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

            For node_idx=2, replacement_token=None:

            0 MethodDeclaration (0)
            ├── 1 parameters (1)
            │   └── 9 FormalParameter (3)
            │       ├── 10 type (4)
            │       │   └── 11 ReferenceType (5)
            │       │       └── 12 name (6)
            │       │           └── 13 File (7)
            │       └── 14 name (8)
            └── 15 body (9)


            The original tensors have been:
            nodes:        [  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15]
            parents:       [ -1,  0,  1,  2,  3,  4,  5,  2,  7,  1,  9, 10, 11, 12,  9,  0]
            #descendants:  [ 14, 13,  6,  3,  2,  1,  0,  1,  0,  5,  3,  2,  1,  0,  0,  0]
            #children:     [  2,  2,  2,  1,  1,  1,  0,  1,  0,  2,  1,  1,  1,  0,  0,  0]

            This method will return the following tensors:
            nodes:        [  0,  1, 99,  9, 10, 11, 12, 13, 14, 15]
            parents:      [ -1,  0,  1,  1,  3,  4,  5,  6,  3,  0]
            #descendants: [  9,  7,  0,  5,  3,  2,  1,  0,  0,  0]
            #children:    [  2,  2,  0,  2,  1,  1,  1,  0,  0,  0]
            """
        return tensortree.delete_subtree(self, node_idx, replacement_token)

    def delete_children(self, node_idx: int, replacement_token: Optional[Any] = None):
        return tensortree.delete_subtree(self, node_idx, replacement_token)

    def swap(self, node_idx: int, other_node_idx: int):
        return tensortree.swap(self, node_idx, other_node_idx)

    def insert_child(self, parent_idx: int, node_data: Any, right_sibling_idx: Optional[int] = None):
        """ adds a node (or a TensorTree) as a child of node at parent_idx, so that it is the left sibling of
         node at right_sibling_idx. If right_sibling_idx is None then it will be appended as the last child."""

        return tensortree.insert_child(self, parent_idx, node_data, right_sibling_idx)



