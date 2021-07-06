import pytest
import torch


import tensortree
from tensortree import TensorTree, TreeStorage
from tensortree import parents_from_descendants, descendants_from_parents


def test_getitem():
    """
        0
        ├── 1
        │   ├── 2
        │   ├── 3
        │   ├── 4
        │   ├── 5
        │   ╰── 6
        ╰── 7
            ╰── 8
                ├── 9
                ├── 10
                ╰── 11
    """

    parents = [-1, 0, 1, 1, 1, 1, 1, 0, 7, 8, 8, 8]
    tree = tensortree.tree(parents=parents)
    tree.pprint()

    node_idx = 8
    subtree = tree[node_idx]
    subtree.pprint()

    assert subtree.get_node_data(0) == tree.get_node_data(8)
    assert subtree.get_node_data(1) == tree.get_node_data(9)
    assert subtree.get_node_data(2) == tree.get_node_data(10)
    assert subtree.get_node_data(3) == tree.get_node_data(11)

    assert subtree.get_number_of_descendants(0) == tree.get_number_of_descendants(8)
    assert subtree.get_number_of_descendants(1) == tree.get_number_of_descendants(9)
    assert subtree.get_number_of_descendants(2) == tree.get_number_of_descendants(10)
    assert subtree.get_number_of_descendants(3) == tree.get_number_of_descendants(11)

    root = subtree[None]
    root.pprint()

    assert root != tree, "Should be new object"
    assert root.data == tree.data, "Should be same data"

    for node_idx in range(len(tree)):
        subtree = tree[node_idx]

        for i in range(len(subtree)):
            assert subtree.get_node_data(i) == tree.get_node_data(node_idx + i)
