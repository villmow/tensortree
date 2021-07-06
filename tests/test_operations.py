import pytest
import torch


import tensortree
from tensortree import TensorTree, TreeStorage
from tensortree import parents_from_descendants, descendants_from_parents


def get_tree() -> TensorTree:
    parents = [-1, 0, 1, 1, 1, 1, 1, 0, 7, 8, 8, 8]
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

    tree = tensortree.tree(parents=parents)
    tree.pprint()
    return tree


def test_parents_to_descendants():
    tree = get_tree()

    other_parents = parents_from_descendants(tree.data.descendants)
    other_descendants = descendants_from_parents(tree.data.parents)
    assert tree.data.parents.tolist() == other_parents.tolist(), "parents should be the same"
    assert tree.data.descendants.tolist() == other_descendants.tolist(), "descendants should be the same"


def test_tensortree_init():
    tree = get_tree()

    other_tree = tensortree.tree(parents=tree.data.parents)
    assert other_tree.data.parents.tolist() == tree.data.parents.tolist()
    assert other_tree.data.descendants.tolist() == tree.data.descendants.tolist()

    other_tree = tensortree.tree(descendants=tree.data.descendants)
    assert other_tree.data.descendants.tolist() == tree.data.descendants.tolist()
    assert other_tree.data.parents.tolist() == tree.data.parents.tolist()

    other_tree = tensortree.tree(descendants=tree.data.descendants, node_data=tree.data.node_data)
    assert other_tree.data.descendants.tolist() == tree.data.descendants.tolist()
    assert other_tree.data.parents.tolist() == tree.data.parents.tolist()
    assert list(other_tree.data.node_data) == list(tree.data.node_data)

    other_tree.pprint()


def test_tensortree_custom_node_type():
    nodes = [
        {"name": "A", "some_attribute": False},
        {"name": "B", "some_attribute": False},
        {"name": "C", "some_attribute": False},
        {"name": "D", "some_attribute": False},
        {"name": "E", "some_attribute": False},
        {"name": "F", "some_attribute": False},
        {"name": "G", "some_attribute": False},
        {"name": "H", "some_attribute": False},
        {"name": "I", "some_attribute": False},
        {"name": "J", "some_attribute": False},
        {"name": "K", "some_attribute": False},
        {"name": "L", "some_attribute": False},
    ]
    parents = [-1, 0, 1, 1, 1, 1, 1, 0, 7, 8, 8, 8]

    tree = tensortree.tree(parents=parents, node_data=nodes)
    tree.pprint()

    tree.pprint(node_renderer=lambda x: x["name"])

    tree.pprint(max_nodes=5, node_renderer=lambda x: x["name"])
    tree[7].pprint(node_renderer=lambda x: x["name"])


def test_tensortree_siblings():
    # tokens = ["A", "B", "C", "D", "E", "F", "G", "H"]
    parents = [-1, 0, 1, 1, 1, 1, 1, 0, 7, 8, 8, 8]
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

    tree = tensortree.tree(parents=parents)
    tree.pprint()

    #########
    node_idx = 3
    sibs = [s.item() for s in tree.iter_siblings(node_idx=node_idx)]
    assert [2, 4, 5, 6] == sibs
    assert tree.siblings(node_idx=node_idx).tolist() == sibs

    sibs = [s.item() for s in tree.iter_siblings(node_idx=node_idx, include_left_siblings=False)]
    assert [4, 5, 6] == sibs
    assert tree.siblings(node_idx=node_idx, check_left=False).tolist() == sibs

    sibs = [s.item() for s in tree.iter_siblings(node_idx=node_idx, include_right_siblings=False)]
    assert [2] == sibs
    assert tree.siblings(node_idx=node_idx, check_right=False).tolist() == sibs
    #################

    #########
    node_idx = 11
    sibs = [s.item() for s in tree.iter_siblings(node_idx=node_idx)]
    assert [9, 10] == sibs
    assert tree.siblings(node_idx=node_idx).tolist() == sibs

    sibs = [s.item() for s in tree.iter_siblings(node_idx=node_idx, include_left_siblings=False)]
    assert [] == sibs
    assert tree.siblings(node_idx=node_idx, check_left=False).tolist() == sibs

    sibs = [s.item() for s in tree.iter_siblings(node_idx=node_idx, include_right_siblings=False)]
    assert [9, 10] == sibs
    assert tree.siblings(node_idx=node_idx, check_right=False).tolist() == sibs
    #################

    #########
    node_idx = 0
    sibs = [s.item() for s in tree.iter_siblings(node_idx=node_idx)]
    assert [] == sibs
    assert tree.siblings(node_idx=node_idx).tolist() == sibs

    sibs = [s.item() for s in tree.iter_siblings(node_idx=node_idx, include_left_siblings=False)]
    assert [] == sibs
    assert tree.siblings(node_idx=node_idx, check_left=False).tolist() == sibs

    sibs = [s.item() for s in tree.iter_siblings(node_idx=node_idx, include_right_siblings=False)]
    assert [] == sibs
    assert tree.siblings(node_idx=node_idx, check_right=False).tolist() == sibs
    #################

    #########
    node_idx = 10
    sibs = [s.item() for s in tree.iter_siblings(node_idx=node_idx)]
    assert [9, 11] == sibs
    assert tree.siblings(node_idx=node_idx).tolist() == sibs

    sibs = [s.item() for s in tree.iter_siblings(node_idx=node_idx, include_left_siblings=False)]
    assert [11] == sibs
    assert tree.siblings(node_idx=node_idx, check_left=False).tolist() == sibs

    sibs = [s.item() for s in tree.iter_siblings(node_idx=node_idx, include_right_siblings=False)]
    assert [9] == sibs
    assert tree.siblings(node_idx=node_idx, check_right=False).tolist() == sibs
    #################

    #########
    node_idx = 9
    sibs = [s.item() for s in tree.iter_siblings(node_idx=node_idx)]
    assert [10, 11] == sibs
    assert tree.siblings(node_idx=node_idx).tolist() == sibs

    sibs = [s.item() for s in tree.iter_siblings(node_idx=node_idx, include_left_siblings=False)]
    assert [10, 11] == sibs
    assert tree.siblings(node_idx=node_idx, check_left=False).tolist() == sibs

    sibs = [s.item() for s in tree.iter_siblings(node_idx=node_idx, include_right_siblings=False)]
    assert [] == sibs
    assert tree.siblings(node_idx=node_idx, check_right=False).tolist() == sibs
    #################

    #########
    node_idx = 7
    sibs = [s.item() for s in tree.iter_siblings(node_idx=node_idx)]
    assert [1] == sibs
    assert tree.siblings(node_idx=node_idx).tolist() == sibs

    sibs = [s.item() for s in tree.iter_siblings(node_idx=node_idx, include_left_siblings=False)]
    assert [] == sibs
    assert tree.siblings(node_idx=node_idx, check_left=False).tolist() == sibs

    sibs = [s.item() for s in tree.iter_siblings(node_idx=node_idx, include_right_siblings=False)]
    assert [1] == sibs
    assert tree.siblings(node_idx=node_idx, check_right=False).tolist() == sibs
    #################

    #########
    node_idx = 8
    sibs = [s.item() for s in tree.iter_siblings(node_idx=node_idx)]
    assert [] == sibs
    assert tree.siblings(node_idx=node_idx).tolist() == sibs

    sibs = [s.item() for s in tree.iter_siblings(node_idx=node_idx, include_left_siblings=False)]
    assert [] == sibs
    assert tree.siblings(node_idx=node_idx, check_left=False).tolist() == sibs

    sibs = [s.item() for s in tree.iter_siblings(node_idx=node_idx, include_right_siblings=False)]
    assert [] == sibs
    assert tree.siblings(node_idx=node_idx, check_right=False).tolist() == sibs
    #################

    node_idx = 12
    with pytest.raises(IndexError):
        _ = [s.item() for s in tree.iter_siblings(node_idx=node_idx)]

    with pytest.raises(IndexError):
        _ = tree.siblings(node_idx=node_idx).tolist()


def test_tensortree_has_sibling():
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

    node_idx = 0
    assert not tree.has_sibling(node_idx=node_idx)
    assert not tree.has_sibling(node_idx=node_idx, check_right=False)
    assert not tree.has_sibling(node_idx=node_idx, check_left=False)

    node_idx = 1
    assert tree.has_sibling(node_idx=node_idx)
    assert not tree.has_sibling(node_idx=node_idx, check_right=False)
    assert tree.has_sibling(node_idx=node_idx, check_left=False)

    node_idx = 2
    assert tree.has_sibling(node_idx=node_idx)
    assert not tree.has_sibling(node_idx=node_idx, check_right=False)
    assert tree.has_sibling(node_idx=node_idx, check_left=False)

    node_idx = 3
    assert tree.has_sibling(node_idx=node_idx)
    assert tree.has_sibling(node_idx=node_idx, check_right=False)
    assert tree.has_sibling(node_idx=node_idx, check_left=False)

    node_idx = 4
    assert tree.has_sibling(node_idx=node_idx)
    assert tree.has_sibling(node_idx=node_idx, check_right=False)
    assert tree.has_sibling(node_idx=node_idx, check_left=False)

    node_idx = 5
    assert tree.has_sibling(node_idx=node_idx)
    assert tree.has_sibling(node_idx=node_idx, check_right=False)
    assert tree.has_sibling(node_idx=node_idx, check_left=False)

    node_idx = 6
    assert tree.has_sibling(node_idx=node_idx)
    assert tree.has_sibling(node_idx=node_idx, check_right=False)
    assert not tree.has_sibling(node_idx=node_idx, check_left=False)

    node_idx = 7
    assert tree.has_sibling(node_idx=node_idx)
    assert tree.has_sibling(node_idx=node_idx, check_right=False)
    assert not tree.has_sibling(node_idx=node_idx, check_left=False)

    node_idx = 8
    assert not tree.has_sibling(node_idx=node_idx)
    assert not tree.has_sibling(node_idx=node_idx, check_right=False)
    assert not tree.has_sibling(node_idx=node_idx, check_left=False)

    node_idx = 9
    assert tree.has_sibling(node_idx=node_idx)
    assert not tree.has_sibling(node_idx=node_idx, check_right=False)
    assert tree.has_sibling(node_idx=node_idx, check_left=False)

    node_idx = 10
    assert tree.has_sibling(node_idx=node_idx)
    assert tree.has_sibling(node_idx=node_idx, check_right=False)
    assert tree.has_sibling(node_idx=node_idx, check_left=False)

    node_idx = 11
    assert tree.has_sibling(node_idx=node_idx)
    assert tree.has_sibling(node_idx=node_idx, check_right=False)
    assert not tree.has_sibling(node_idx=node_idx, check_left=False)

    with pytest.raises(IndexError):
        tree.has_sibling(node_idx=12)

    with pytest.raises(IndexError):
        tree.has_sibling(node_idx=12, check_left=False)

    with pytest.raises(IndexError):
        tree.has_sibling(node_idx=12, check_right=False)


def test_tensortree_right_sibling():
    """
    0. 0
    ├── 1. 1
    │   ├── 2. 2
    │   ├── 3. 3
    │   ├── 4. 4
    │   ├── 5. 5
    │   ╰── 6. 6
    ├── 7. 7
    │   ├── 8. 8
    │   │   ├── 9. 9
    │   │   ├── 10. 10
    │   │   ╰── 11. 11
    │   ├── 12. 12
    │   ╰── 13. 13
    ╰── 14. 14
        ╰── 15. 15
    """
    parents = [-1, 0, 1, 1, 1, 1, 1, 0, 7, 8, 8, 8, 7, 7, 0, 14]

    tree = tensortree.tree(parents=parents)
    tree.pprint()

    assert tree.right_sibling(node_idx=0) is None
    assert tree.right_sibling(node_idx=1) == 7
    assert tree.right_sibling(node_idx=2) == 3
    assert tree.right_sibling(node_idx=3) == 4
    assert tree.right_sibling(node_idx=4) == 5
    assert tree.right_sibling(node_idx=5) == 6
    assert tree.right_sibling(node_idx=6) is None
    assert tree.right_sibling(node_idx=7) == 14
    assert tree.right_sibling(node_idx=8) == 12
    assert tree.right_sibling(node_idx=9) == 10
    assert tree.right_sibling(node_idx=10) == 11
    assert tree.right_sibling(node_idx=11) is None
    assert tree.right_sibling(node_idx=12) == 13
    assert tree.right_sibling(node_idx=13) is None
    assert tree.right_sibling(node_idx=14) is None
    assert tree.right_sibling(node_idx=15) is None


def test_tensortree_is_leaf():
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

    leaves = [False] * 2 + [True] * 5 + [False] * 2 + [True] * 3

    for node_idx in range(len(tree)):
        assert leaves[node_idx] == tree.is_leaf(node_idx=node_idx)

    with pytest.raises(IndexError):
        tree.is_leaf(node_idx=-1)
        tree.is_leaf(node_idx=-1)
        tree.is_leaf(node_idx=12)


def test_tensortree_children():
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

    assert tree.children(node_idx=0).tolist() == [1, 7]
    assert tree.children(node_idx=1).tolist() == [2, 3, 4, 5, 6]
    assert tree.children(node_idx=2).tolist() == []
    assert tree.children(node_idx=3).tolist() == []
    assert tree.children(node_idx=4).tolist() == []
    assert tree.children(node_idx=5).tolist() == []
    assert tree.children(node_idx=6).tolist() == []
    assert tree.children(node_idx=7).tolist() == [8]
    assert tree.children(node_idx=8).tolist() == [9, 10, 11]
    assert tree.children(node_idx=9).tolist() == []
    assert tree.children(node_idx=10).tolist() == []
    assert tree.children(node_idx=11).tolist() == []

    # with pytest.raises(IndexError):
    tree.children(node_idx=12).tolist()

    with pytest.raises(IndexError):
        tree.children(node_idx=-12).tolist()


def test_tensortree_next_node_not_in_branch():
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

    assert tree.next_node_not_in_branch(node_idx=0) is None
    assert tree.next_node_not_in_branch(node_idx=1) == 7
    assert tree.next_node_not_in_branch(node_idx=2) == 3
    assert tree.next_node_not_in_branch(node_idx=3) == 4
    assert tree.next_node_not_in_branch(node_idx=4) == 5
    assert tree.next_node_not_in_branch(node_idx=5) == 6
    assert tree.next_node_not_in_branch(node_idx=6) == 7
    assert tree.next_node_not_in_branch(node_idx=7) is None
    assert tree.next_node_not_in_branch(node_idx=8) is None
    assert tree.next_node_not_in_branch(node_idx=9) == 10
    assert tree.next_node_not_in_branch(node_idx=10) == 11
    assert tree.next_node_not_in_branch(node_idx=11) is None

    with pytest.raises(IndexError):
        tree.next_node_not_in_branch(node_idx=12)


def test_tensortree_parent():
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

    assert tree.get_parent(node_idx=0) is None
    for n_idx in range(1, len(parents)):
        assert tree.get_parent(node_idx=n_idx) == parents[n_idx]

    with pytest.raises(IndexError):
        tree.get_parent(node_idx=12)


def test_tensortree_to_string():
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
    # FIXME
    parents = [-1, 0, 1, 1, 1, 1, 1, 0, 7, 8, 8, 8]

    tree = tensortree.tree(parents=parents)
    tree.pprint()
    for i in range(len(tree)):
        i += 1
        print("#########", i, "##########")
        tree.pprint(max_nodes=i)
        print("#########")
    print(str(tree))


def test_tensortree_replace_branch_string():
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
    tokens = [t for t in "ABCDEFGHIJKL"]
    parents = [-1, 0, 1, 1, 1, 1, 1, 0, 7, 8, 8, 8]
    tree = tensortree.tree(node_data=tokens, parents=parents)
    tree.pprint()
    assert len(tree) == 12

    other_tree = tree.delete_node(1, replacement_token="Qwertz")
    other_tree.pprint()
    assert len(other_tree) == 7


def test_tensortree_getitem():
    """
    0. A
    ├── 1. B
    │   ├── 2. C
    │   ├── 3. D
    │   ├── 4. E
    │   ├── 5. F
    │   ╰── 6. G
    ╰── 7. H
        ╰── 8. I
            ├── 9. J
            ├── 10. K
            ╰── 11. L
    """
    tokens = [t for t in "ABCDEFGHIJKL"]
    parents = [-1, 0, 1, 1, 1, 1, 1, 0, 7, 8, 8, 8]
    tree = tensortree.tree(node_data=tokens, parents=parents)

    for subtree in tree:
        print("-"*20)
        subtree.pprint()
        print(subtree.root_idx, len(subtree))
        for r in range(len(parents)):
            if r < subtree.root_idx or r > (subtree.root_idx + len(subtree)):
                with pytest.raises(IndexError):
                    print("retrieving index", r, "from subtree")
                    another = subtree[r]

    ##################################
    subtree = tree[7]
    subtree.pprint()
    assert len(subtree) == 5
    assert subtree.get_parent(7) is None

    #fixme fix this test


def test_tensortree_delete_node():
    """
    0. 0
    ├── 1. 1
    │   ├── 2. 2
    │   ├── 3. 3
    │   ├── 4. 4
    │   ├── 5. 5
    │   ╰── 6. 6
    ├── 7. 7
    │   ├── 8. 8
    │   │   ├── 9. 9
    │   │   ├── 10. 10
    │   │   ╰── 11. 11
    │   ├── 12. 12
    │   ╰── 13. 13
    ╰── 14. 14
        ╰── 15. 15
    """
    parents = [-1, 0, 1, 1, 1, 1, 1, 0, 7, 8, 8, 8, 7, 7, 0, 14]

    tree = tensortree.tree(parents=parents)
    tree.pprint()

    other_tree = tree.delete_node(1)
    other_tree.pprint()
    assert len(other_tree) == len(tree) - 6
    assert len(tree) == len(parents), "original tree should stay the same"

    other_tree = tree.delete_node(7)
    other_tree.pprint()
    assert len(other_tree) == len(tree) - 7

    other_tree = tree.delete_node(14)
    other_tree.pprint()
    assert len(other_tree) == len(tree) - 2

    other_tree = tree.delete_node(4)
    other_tree.pprint()
    assert len(other_tree) == len(tree) - 1

    # delete root returns empty tree
    other_tree = tree.delete_node(0)
    assert len(other_tree) == 0
    print(other_tree)
    other_tree.pprint()


def test_tensortree_delete_node_with_replacement():
    """
    0. 0
    ├── 1. 1
    │   ├── 2. 2
    │   ├── 3. 3
    │   ├── 4. 4
    │   ├── 5. 5
    │   ╰── 6. 6
    ├── 7. 7
    │   ├── 8. 8
    │   │   ├── 9. 9
    │   │   ├── 10. 10
    │   │   ╰── 11. 11
    │   ├── 12. 12
    │   ╰── 13. 13
    ╰── 14. 14
        ╰── 15. 15
    """
    parents = [-1, 0, 1, 1, 1, 1, 1, 0, 7, 8, 8, 8, 7, 7, 0, 14]

    tree = tensortree.tree(parents=parents)
    tree.pprint()

    other_tree = tree.delete_node(1, replacement_token=999)
    other_tree.pprint()
    assert len(other_tree) == len(tree) - 6 + 1
    assert len(tree) == len(parents), "original tree should stay the same"

    other_tree = tree.delete_node(7, replacement_token=999)
    other_tree.pprint()
    assert len(other_tree) == len(tree) - 7 + 1

    other_tree = tree.delete_node(14, replacement_token=999)
    other_tree.pprint()
    assert len(other_tree) == len(tree) - 2 + 1

    other_tree = tree.delete_node(4, replacement_token=999)
    other_tree.pprint()
    assert len(other_tree) == len(tree) - 1 + 1

    # delete root returns empty tree
    other_tree = tree.delete_node(0, replacement_token=999)
    assert len(other_tree) == 0 + 1
    other_tree.pprint()


def test_tensortree_swap():
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

    print("#" * 50)
    other_tree = tree.swap(1, 8)
    other_tree.pprint()

    # check if we can convert between parents and descendants
    assert other_tree.data.descendants.tolist() == descendants_from_parents(other_tree.data.parents).tolist()
    assert other_tree.data.parents.tolist() == parents_from_descendants(other_tree.data.descendants).tolist()

    print("#" * 50)
    other_tree = tree.swap(2, 8)
    other_tree.pprint()

    # check if we can convert between parents and descendants
    assert other_tree.data.descendants.tolist() == descendants_from_parents(other_tree.data.parents).tolist()
    assert other_tree.data.parents.tolist() == parents_from_descendants(other_tree.data.descendants).tolist()


def test_tensortree_insert_child_int():
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


    for i in range(len(tree)):
        other_tree = tree.insert_child(i, 666)
        assert other_tree.data.descendants.tolist() == descendants_from_parents(other_tree.data.parents).tolist()
        assert other_tree.data.parents.tolist() == parents_from_descendants(other_tree.data.descendants).tolist()
        other_tree.pprint()

    with pytest.raises(IndexError):
        other_tree = tree.insert_child(-1, 666)
        other_tree.pprint()

    with pytest.raises(IndexError):
        other_tree = tree.insert_child(12, 666)
        other_tree.pprint()


def test_tensortree_insert_child_int_with_right_sibling():
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

    other_tree = tree.insert_child(0, 666, right_sibling_idx=1)
    other_tree.pprint()
    assert other_tree.data.descendants.tolist() == descendants_from_parents(other_tree.data.parents).tolist()
    assert other_tree.data.parents.tolist() == parents_from_descendants(other_tree.data.descendants).tolist()

    other_tree = tree.insert_child(1, 666, right_sibling_idx=2)
    other_tree.pprint()
    assert other_tree.data.descendants.tolist() == descendants_from_parents(other_tree.data.parents).tolist()
    assert other_tree.data.parents.tolist() == parents_from_descendants(other_tree.data.descendants).tolist()

    other_tree = tree.insert_child(1, 666, right_sibling_idx=3)
    other_tree.pprint()
    assert other_tree.data.descendants.tolist() == descendants_from_parents(other_tree.data.parents).tolist()
    assert other_tree.data.parents.tolist() == parents_from_descendants(other_tree.data.descendants).tolist()

    other_tree = tree.insert_child(1, 666, right_sibling_idx=4)
    other_tree.pprint()
    assert other_tree.data.descendants.tolist() == descendants_from_parents(other_tree.data.parents).tolist()
    assert other_tree.data.parents.tolist() == parents_from_descendants(other_tree.data.descendants).tolist()

    other_tree = tree.insert_child(1, 666, right_sibling_idx=5)
    other_tree.pprint()
    assert other_tree.data.descendants.tolist() == descendants_from_parents(other_tree.data.parents).tolist()
    assert other_tree.data.parents.tolist() == parents_from_descendants(other_tree.data.descendants).tolist()

    other_tree = tree.insert_child(1, 666, right_sibling_idx=6)
    other_tree.pprint()
    assert other_tree.data.descendants.tolist() == descendants_from_parents(other_tree.data.parents).tolist()
    assert other_tree.data.parents.tolist() == parents_from_descendants(other_tree.data.descendants).tolist()

    with pytest.raises(IndexError):
        other_tree = tree.insert_child(1, 666, right_sibling_idx=7)
        other_tree.pprint()


def test_tensortree_insert_child_tree_right_sibling():
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

    print("will attach a subtree to this tree:")
    tree = tensortree.tree(parents=parents)
    tree.pprint()

    print("this subtree:")
    subtree = tree[7].detach()
    subtree.pprint()

    # always append as the first child
    for parent_idx in range(len(tree)):
        print("#" * 50)
        print(parent_idx)
        first_child = parent_idx + 1 if not tree.is_leaf(parent_idx) else None
        other_tree = tree.insert_child(parent_idx, subtree, right_sibling_idx=first_child)
        assert other_tree.data.descendants.tolist() == descendants_from_parents(other_tree.data.parents).tolist()
        assert other_tree.data.parents.tolist() == parents_from_descendants(other_tree.data.descendants).tolist()
        other_tree.pprint()



def test_tensortree_insert_child_tree():
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

    print("will attach a subtree to this tree:")
    tree = tensortree.tree(parents=parents)
    tree.pprint()

    print("this subtree:")
    subtree = tree[7].detach()
    subtree.pprint()

    for i in range(len(tree)):
        other_tree = tree.insert_child(i, subtree)
        assert other_tree.data.descendants.tolist() == descendants_from_parents(other_tree.data.parents).tolist()
        assert other_tree.data.parents.tolist() == parents_from_descendants(other_tree.data.descendants).tolist()
        other_tree.pprint()

    with pytest.raises(IndexError):
        other_tree = tree.insert_child(-1, subtree)
        other_tree.pprint()

    with pytest.raises(IndexError):
        other_tree = tree.insert_child(12, subtree)
        other_tree.pprint()


def test_tensortree_append_to_only_root():
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

    # create single token tree and append the other tree to set a new root
    print("will attach a subtree to this tree:")
    tree = tensortree.tree(parents=parents)
    tree.pprint()

    print("this subtree:")
    only_root = tensortree.tree(node_data=[999], parents=[-1], descendants=[0])
    other_tree = only_root.insert_child(0, tree)

    other_tree.pprint()


def test_tensortree_height():
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

    # create single token tree and append the other tree to set a new root
    print("will attach a subtree to this tree:")
    tree = tensortree.tree(parents=parents)
    tree.pprint()

    from tensortree import mask_layer
    heights = torch.zeros_like(tree.data.parents)

    for i, layer_mask in enumerate(mask_layer(tree.node_incidence_matrix())):
        heights[layer_mask] = i

    print(heights)

    node_data = [f"height: {h.item()}" for h in heights]
    tree = tensortree.tree(parents=parents, node_data=heights)
    tree.pprint()

    print("this subtree:")
    only_root = tensortree.tree(node_data=[999], parents=[-1], descendants=[0])
    other_tree = only_root.insert_child(0, tree)

    other_tree.pprint()


def test_tensortree_leaves_per_subtree():
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

    leaves = tree.leaves_mask()
    print(torch.cumsum(leaves, dim=-1))


def test_tensortree_insert_child_tree_custom_node_types():
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
    nodes = [
        {"name": "A", "some_attribute": False},
        {"name": "B", "some_attribute": False},
        {"name": "C", "some_attribute": False},
        {"name": "D", "some_attribute": False},
        {"name": "E", "some_attribute": False},
        {"name": "F", "some_attribute": False},
        {"name": "G", "some_attribute": False},
        {"name": "H", "some_attribute": False},
        {"name": "I", "some_attribute": False},
        {"name": "J", "some_attribute": False},
        {"name": "K", "some_attribute": False},
        {"name": "L", "some_attribute": False},
    ]
    parents = [-1, 0, 1, 1, 1, 1, 1, 0, 7, 8, 8, 8]

    print("will attach a subtree to this tree:")
    tree = tensortree.tree(parents=parents, node_data=nodes)
    tree.pprint(node_renderer=lambda x: x["name"])

    print("this subtree:")
    subtree = tree[7].detach()
    subtree.pprint(node_renderer=lambda x: x["name"])

    for i in range(len(tree)):
        other_tree = tree.insert_child(i, subtree)
        assert other_tree.data.descendants.tolist() == descendants_from_parents(other_tree.data.parents).tolist()
        assert other_tree.data.parents.tolist() == parents_from_descendants(other_tree.data.descendants).tolist()
        other_tree.pprint(node_renderer=lambda x: x["name"])

    with pytest.raises(IndexError):
        other_tree = tree.insert_child(-1, subtree)
        other_tree.pprint(node_renderer=lambda x: x["name"])

    with pytest.raises(IndexError):
        other_tree = tree.insert_child(12, subtree)
        other_tree.pprint(node_renderer=lambda x: x["name"])


def test_tensortree_merge_order():
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

    merge_order = [4, 2, 1, 1, 1, 1, 1, 3, 2, 1, 1, 1]


    d = tree.descendants
    import torch
    # print(torch.cumsum(di, -1))
    # print(di)

    processed = torch.zeros(d.shape, dtype=torch.int)
    print(processed)
    i = 1
    ni = tree.node_incidence_matrix()
    r = torch.arange(len(tree))
    while (processed == 0).any():
        leaves = ni.sum(0) == 1
        ind = r[leaves]
        r = r[~leaves]
        processed[ind] = i
        ni = ni[~leaves][:,~leaves]
        i += 1
        print(processed)
        print(ni)


def test_adjacency_matrix():
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
    parents = [-1,  0,  1,  2,  2,  1,  0,  6]
    tree = tensortree.tree(parents=parents)
    tree.pprint()

    for node_idx in range(len(tree)):
        children_mask_with_parents = (tree.parents == node_idx)
        children_mask_adjacency = tree.adjacency_matrix()[node_idx]
        print(children_mask_adjacency.byte())
        print(children_mask_with_parents.byte())
        print("-------------")
        assert torch.all(children_mask_adjacency == children_mask_with_parents), "Both arrays should be equal"