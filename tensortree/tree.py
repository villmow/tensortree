from dataclasses import dataclass
from typing import Sequence, Any, Union, List, Optional, Tuple, Generator, Callable, Literal

import numpy as np
import torch

import tensortree
from tensortree.render import Style, ContRoundStyle, format_tree
from tensortree.utils import to_torch, validate_index
from functools import lru_cache


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

    @classmethod
    def from_node_incidences(cls, node_incidences: torch.Tensor, node_data: Optional[Sequence[LabelType]] = None):
        descendants = tensortree.descendants_from_node_incidences(node_incidences)
        return cls.from_array(descendants=descendants, node_data=node_data)

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

    def __len__(self):
        """ The number of nodes in this tree. """
        return self.__len

    def __str__(self):
        return self.pformat(max_nodes=10)

    @validate_index(allow_none=True)
    def __getitem__(self, node_idx: Union[int, torch.Tensor, np.ndarray, Tuple[int, ...], None]):
        """ Will returns a view of the node at node_idx"""
        if node_idx is None and node_idx != self.root_idx:
            return TensorTree(self.data)  # take me back to the very root

        node_idx = self._to_global_idx(node_idx)

        if node_idx == self.root_idx:
            return self

        return TensorTree(self.data, node_idx)

    def detach(self):
        """ Returns a new tree rooted at self.root_idx """
        from copy import deepcopy
        return self.from_array(
            parents=self.parents.clone(),
            descendants=self.descendants.clone(),
            node_data=deepcopy(self.node_data)
        )

    def is_subtree(self) -> bool:
        """
        Is this tree instance part of a bigger tree?

        You can retrieve the original tree using
        >>> root = subtree[None]
        """

        return self.root_idx > 0

    def is_descendant_of(self, node_ancestor_idx: int, node_descendant_idx: int) -> bool:
        """
        Is one node a descendant of another node?

        :param node_ancestor_idx: The node closer to the root
        :param node_descendant_idx: The descendant
        :return:
        """
        return node_ancestor_idx < node_descendant_idx < (
                node_ancestor_idx + self.get_number_of_descendants(node_ancestor_idx)
        )

    # helpers for individual nodes
    @validate_index
    def get_node_data(self, node_idx: Union[int, torch.Tensor]) -> Any:
        """
        Returns the data of a node.

        :param node_idx: The index of the node.
        """
        node_idx = self._to_global_idx(node_idx)
        return self.data.node_data[node_idx]

    @validate_index
    def get_number_of_descendants(self, node_idx: Union[int, torch.Tensor]) -> torch.Tensor:
        """
        Returns the amount of descendants of a node.

        :param node_idx:
        :return:
        """
        node_idx = self._to_global_idx(node_idx)
        return self.data.descendants[node_idx]

    @validate_index
    def get_parent(self, node_idx: Union[int, torch.Tensor]) -> Optional[int]:
        """ Returns the parent idx for this node or None if node_idx is root.

        :param node_idx: Node_idx refers to the index of the node in the original tree.
        :return:
        """
        node_idx = self._to_global_idx(node_idx)

        # return None for root
        if node_idx == self.root_idx:
            return

        return self.data.parents[node_idx]

    @property
    def descendants(self) -> torch.Tensor:
        """
        Returns the relevant subset of descendants for this subtree. This will return
         a slice of the data. Changing this object will change the storage and may
         invalidate the tree. This is not checked.
        """
        if self.root_idx == 0:
            return self.data.descendants  # return full array
        else:
            return self.data.descendants[self.root_idx: self.end]  # return slice

    @property
    def node_data(self) -> Sequence[LabelType]:
        """
        Returns the relevant subset of node_data for this tree. This will return
         a slice of the data. Changing this object will change the storage and may
         invalidate the tree. This is not checked.

        """
        if self.root_idx == 0:
            return self.data.node_data
        else:
            return self.data.node_data[self.root_idx: self.end]

    @property
    def parents(self) -> torch.Tensor:
        """
        Returns the relevant subset of parents for this subtree. This returns the underlying
         storage only if it is root. Otherwise will create a new array for this subtree.

        """
        if self.root_idx == 0:
            return self.data.parents

        # compute parents
        if getattr(self, '__subtree_parents', None) is None:
            parents = self.data.parents[self.root_idx: self.end] - self.root_idx
            parents[0] = -1
            self.__subtree_parents = parents  # cache parents for this subtree

        return self.__subtree_parents

    @lru_cache(maxsize=1)
    def node_incidence_matrix(self) -> torch.BoolTensor:
        """ Returns the node incidence matrix for this tree"""
        return tensortree.node_incidence_matrix(self.descendants)

    @lru_cache(maxsize=1)
    def adjacency_matrix(
            self, directed: bool = True, direction_up: bool = False, loop_root: bool = False) -> torch.BoolTensor:
        """
         Returns the adjacency matrix for this tree.

        :param directed: Should the adjacency matrix be directed?
        :param direction_up: Per default the root points at its children.
        :param loop_root: Should root have a loop.
        :return:
        """
        return tensortree.adjacency_matrix(
            parents=self.parents,
            directed=directed, direction_up=direction_up,
            loop_root=loop_root
        )

    def ancestral_matrix(self) -> torch.LongTensor:
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

        :return:
        """

        return tensortree.ancestral_matrix(self.node_incidence_matrix())

    def movements(self) -> torch.LongTensor:
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
        return tensortree.movements(self.node_incidence_matrix())

    def distances(self) -> torch.LongTensor:
        """
        Computes distances between nodes when the tree is seen as undirected.

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

        Will return the following distances between nodes.
        tensor([[0, 1, 1, 2, 2, 3],
                [1, 0, 2, 3, 3, 4],
                [1, 2, 0, 1, 1, 2],
                [2, 3, 1, 0, 2, 3],
                [2, 3, 1, 2, 0, 1],
                [3, 4, 2, 3, 1, 0]])

        Will always return a LongTensor.
        """
        return tensortree.distances(self.node_incidence_matrix())

    def least_common_ancestors(self):
        """
        Return a least common ancestor matrix for this tree.

        :return:
        """
        return tensortree.least_common_ancestors(self.node_incidence_matrix())

    def levels(self) -> torch.LongTensor:
        """
        Computes the level of each node (i.e. the number of edges from a node to the root).
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
        return tensortree.levels(self.node_incidence_matrix())

    @validate_index
    def next_node_not_in_branch(self, node_idx: Union[int, torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Return the next node, that is not part of the branch of node at node_idx.

        Can be a sibling or any other node that follows after this branch.

        :param node_idx: Can be either an integer or a single value tensor
        :return:
        """
        next_node_idx = (node_idx + self.get_number_of_descendants(node_idx) + 1)
        return next_node_idx if next_node_idx < len(self) else None

    @validate_index
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

        current_node_idx = 0
        for child_number in tree_position:
            current_node_idx = get_nth_child(current_node_idx, child_number)

        return current_node_idx

    def heights(self):
        """ Depth of each subtree"""
        heights = torch.zeros_like(self.descendants)

        for i, layer_mask in enumerate(tensortree.mask_layer(self.node_incidence_matrix())):
            heights[layer_mask] = i

        return heights

    # subtree mask
    @validate_index
    def subtree_mask(self, node_idx: Union[int, torch.Tensor]) -> torch.BoolTensor:
        """ returns a mask which selects the relevant nodes of a subtree from the array """

        start = node_idx
        end = self.next_node_not_in_branch(node_idx)

        mask = torch.zeros_like(self.descendants, dtype=torch.bool)
        mask[start:end] = True
        return mask

    # leaves
    @validate_index
    def is_leaf(self, node_idx: Union[int, torch.Tensor]) -> bool:
        return self.get_number_of_descendants(node_idx) == 0

    def leaves_mask(self) -> torch.BoolTensor:
        """ Returns a boolean mask for all leaf nodes in this tree """
        return self.descendants == 0

    def leaf_indices(self) -> torch.Tensor:
        return self.leaves_mask().nonzero(as_tuple=False).squeeze(-1)

    # children
    @validate_index
    def iter_children(self, node_idx: Union[int, torch.Tensor]) -> Generator[torch.Tensor, None, None]:
        """
        Iters over the children of a node with a specific index in a tree.

        :param node_idx: Node to iterate over the children
        """
        branch_end = self.next_node_not_in_branch(node_idx)  # end of subtree of node_idx

        if branch_end is None:
            branch_end = len(self)

        # next child is at the next position in the descendants array
        next_child = node_idx + 1

        # are we still in the subtree
        while next_child is not None and next_child < branch_end:
            yield next_child
            next_child = self.next_node_not_in_branch(next_child)

    @validate_index
    def children_mask(self, node_idx: Union[int, torch.Tensor]) -> torch.BoolTensor:
        """
        Returns a mask over the child indices of a node.

        :param node_idx: Node to get the children
        """
        return self.adjacency_matrix()[node_idx]

    @validate_index
    def children(self, node_idx: Union[int, torch.Tensor]) -> torch.Tensor:
        """
        Returns children indices of a node at a specific index in a tree.

        :param node_idx: Node to get the children
        """
        return self.children_mask(node_idx).nonzero(as_tuple=False).squeeze(-1)

    # siblings
    @validate_index
    def iter_siblings(self, node_idx: Union[int, torch.Tensor], include_left_siblings: bool = True, include_right_siblings: bool = True) -> Generator[torch.Tensor, None, None]:
        """ Node indices with the same parent. The node at node_idx is excluded.

        set include_left_siblings to False to only output siblings which are to the right of node at node_idx.
        set include_right_siblings to False  to only output siblings which are to the left of node at node_idx.
        """
        if not (include_left_siblings or include_right_siblings):
            raise ValueError("Left or right must be specified.")

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

    @validate_index
    def right_sibling(self, node_idx: Union[int, torch.Tensor]) -> Optional[torch.Tensor]:
        next_node = self.next_node_not_in_branch(node_idx)
        if next_node is not None:
            if self.get_parent(next_node) == self.get_parent(node_idx):
                return next_node

    @validate_index
    def siblings_mask(self, node_idx: Union[int, torch.Tensor], include_left_siblings: bool = True, include_right_siblings: bool = True) -> torch.BoolTensor:
        """ Node indices with the same parent. The node at node_idx is excluded.

        set include_left_siblings to False to only output siblings which are to the right of node at node_idx.
        set right to False  to only output siblings which are to the left of node at node_idx.
        """
        if not (include_left_siblings or include_right_siblings):
            raise ValueError("Left or right must be specified.")

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

    @validate_index
    def siblings(self, node_idx: Union[int, torch.Tensor], check_left: bool = True, check_right: bool = True) -> torch.Tensor:
        return self.siblings_mask(node_idx, check_left, check_right).nonzero(as_tuple=False).squeeze(-1)

    @validate_index
    def has_sibling(self, node_idx: Union[int, torch.Tensor], check_left: bool = True, check_right: bool = True) -> bool:
        """
        Is there a sibling to this node?

        :param node_idx:
        :param check_left: look left for siblings (set to False to only check for right siblings)
        :param check_right: look right for siblings (set to False to only check for left siblings)
        :return:
        """
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
    @validate_index
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

    @validate_index
    def _to_global_idx(self, node_idx: Union[int, torch.Tensor]) -> Union[int, torch.Tensor]:
        """ Transfers a node_idx inside a subtree view to a global node_idx """
        return node_idx + self.root_idx

    # functions below return modify and return a new tree
    @validate_index
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

    @validate_index
    def delete_children(self, node_idx: int, replacement_token: Optional[Any] = None):
        return tensortree.delete_children(self, node_idx, replacement_token)

    @validate_index
    def swap(self, node_idx: int, other_node_idx: int):
        return tensortree.swap(self, node_idx, other_node_idx)

    @validate_index
    def insert_child(self, parent_idx: int, node_data: Any, right_sibling_idx: Optional[int] = None):
        """ adds a node (or a TensorTree) as a child of node at parent_idx, so that it is the left sibling of
         node at right_sibling_idx. If right_sibling_idx is None then it will be appended as the last child."""

        return tensortree.insert_child(self, parent_idx, node_data, right_sibling_idx)



