import pytest

import autocoder.tree as actree
from autocoder.tokenizer.code_bpe import TreeSentencepieceBPE
from autocoder.tokenizer.code_tokenizer import CodeTokenizer
from autocoder.tree import new_tree


def get_tokenizer():
    return CodeTokenizer.from_kwargs(
        ct_lang="java",
        ct_to_closest_ascii=True,
        ct_keep_comments=True,
        ct_clean_comments=True,
        ct_replace_whitespace=True,
    )


def get_bpe():
    return TreeSentencepieceBPE.from_kwargs(
        bpe_sentencepiece_model="/home/johannes/projects/semcode/autocoder/tests/assets/bpe.16000.model",
        bpe_mode="leaves"
    )


def get_code():
    return """
        import java.swing;
        import java.swing;
        import java.swing;
        import java.swing;

        /*
         * Java implementation of the approach
         */ 
        public class GFG { 

            // Function that returns true if 
            // str is a palindrome 
            static boolean isPalindrome(String str) 
            { 

                // Pointers pointing to the beginning 
                // and the end of the string 
                int i = 0, j = str.length() - 1; 

                // While there are characters toc compare 
                while (i < j) { 

                    // If there is a mismatch 
                    if (str.charAt(i) != str.charAt(j)) 
                        return false; 

                    // Increment first pointer and 
                    // decrement the other 
                    i++; 
                    j--; 
                } 

                // Given string is a palindrome 
                return true; 
            } 

            // Driver code 
            public static void main(String[] args) 
            { 
                String str = "geeks is a palindrome"; 

                if (isPalindrome(str)) 
                    System.out.print("Yes"); 
                else
                    System.out.print("No"); 
            } 
        } 
        """


def get_tree():
    code = get_code()
    tokenizer = get_tokenizer()

    tree = tokenizer.code_to_tree(code)
    tokens, parents = tokenizer.encode(code)

    descendants = tree.all_descendants()

    return tokens, parents, descendants


def test_parents_to_descendants():
    tokens, parents, descendants = get_tree()

    other_parents = actree.parents_from_descendants(descendants)
    other_descendants = actree.descendants_from_parents(parents)
    assert list(parents) == list(other_parents), "parents should be the same"
    assert list(descendants) == list(other_descendants), "descendants should be the same"


def test_tensortree_init():
    tokens, parents, descendants = get_tree()

    tree = new_tree(parents=parents)
    assert tree.data.parents.tolist() == list(parents)
    assert tree.data.descendants.tolist() == list(descendants)

    tree = new_tree(descendants=descendants)
    assert tree.data.descendants.tolist() == list(descendants)
    assert tree.data.parents.tolist() == list(parents)

    tree = new_tree(descendants=descendants, labels=tokens)
    assert tree.data.descendants.tolist() == list(descendants)
    assert tree.data.parents.tolist() == list(parents)
    assert list(tree.data.labels) == list(tokens)

    tree.pprint()


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

    tree = new_tree(parents=parents, labels=nodes)
    tree.pprint()

    tree.pprint(node_renderer=lambda x: x["name"])

    tree.pprint(max_nodes=5, node_renderer=lambda x: x["name"])


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

    tree = new_tree(parents=parents)
    tree.pprint()

    #########
    node_idx = 3
    sibs = [s.item() for s in tree.iter_siblings(node_idx=node_idx)]
    assert [2, 4, 5, 6] == sibs
    assert tree.siblings(node_idx=node_idx).tolist() == sibs

    sibs = [s.item() for s in tree.iter_siblings(node_idx=node_idx, left=False)]
    assert [4, 5, 6] == sibs
    assert tree.siblings(node_idx=node_idx, left=False).tolist() == sibs

    sibs = [s.item() for s in tree.iter_siblings(node_idx=node_idx, right=False)]
    assert [2] == sibs
    assert tree.siblings(node_idx=node_idx, right=False).tolist() == sibs
    #################

    #########
    node_idx = 11
    sibs = [s.item() for s in tree.iter_siblings(node_idx=node_idx)]
    assert [9, 10] == sibs
    assert tree.siblings(node_idx=node_idx).tolist() == sibs

    sibs = [s.item() for s in tree.iter_siblings(node_idx=node_idx, left=False)]
    assert [] == sibs
    assert tree.siblings(node_idx=node_idx, left=False).tolist() == sibs

    sibs = [s.item() for s in tree.iter_siblings(node_idx=node_idx, right=False)]
    assert [9, 10] == sibs
    assert tree.siblings(node_idx=node_idx, right=False).tolist() == sibs
    #################

    #########
    node_idx = 0
    sibs = [s.item() for s in tree.iter_siblings(node_idx=node_idx)]
    assert [] == sibs
    assert tree.siblings(node_idx=node_idx).tolist() == sibs

    sibs = [s.item() for s in tree.iter_siblings(node_idx=node_idx, left=False)]
    assert [] == sibs
    assert tree.siblings(node_idx=node_idx, left=False).tolist() == sibs

    sibs = [s.item() for s in tree.iter_siblings(node_idx=node_idx, right=False)]
    assert [] == sibs
    assert tree.siblings(node_idx=node_idx, right=False).tolist() == sibs
    #################

    #########
    node_idx = 10
    sibs = [s.item() for s in tree.iter_siblings(node_idx=node_idx)]
    assert [9, 11] == sibs
    assert tree.siblings(node_idx=node_idx).tolist() == sibs

    sibs = [s.item() for s in tree.iter_siblings(node_idx=node_idx, left=False)]
    assert [11] == sibs
    assert tree.siblings(node_idx=node_idx, left=False).tolist() == sibs

    sibs = [s.item() for s in tree.iter_siblings(node_idx=node_idx, right=False)]
    assert [9] == sibs
    assert tree.siblings(node_idx=node_idx, right=False).tolist() == sibs
    #################

    #########
    node_idx = 9
    sibs = [s.item() for s in tree.iter_siblings(node_idx=node_idx)]
    assert [10, 11] == sibs
    assert tree.siblings(node_idx=node_idx).tolist() == sibs

    sibs = [s.item() for s in tree.iter_siblings(node_idx=node_idx, left=False)]
    assert [10, 11] == sibs
    assert tree.siblings(node_idx=node_idx, left=False).tolist() == sibs

    sibs = [s.item() for s in tree.iter_siblings(node_idx=node_idx, right=False)]
    assert [] == sibs
    assert tree.siblings(node_idx=node_idx, right=False).tolist() == sibs
    #################

    #########
    node_idx = 7
    sibs = [s.item() for s in tree.iter_siblings(node_idx=node_idx)]
    assert [1] == sibs
    assert tree.siblings(node_idx=node_idx).tolist() == sibs

    sibs = [s.item() for s in tree.iter_siblings(node_idx=node_idx, left=False)]
    assert [] == sibs
    assert tree.siblings(node_idx=node_idx, left=False).tolist() == sibs

    sibs = [s.item() for s in tree.iter_siblings(node_idx=node_idx, right=False)]
    assert [1] == sibs
    assert tree.siblings(node_idx=node_idx, right=False).tolist() == sibs
    #################

    #########
    node_idx = 8
    sibs = [s.item() for s in tree.iter_siblings(node_idx=node_idx)]
    assert [] == sibs
    assert tree.siblings(node_idx=node_idx).tolist() == sibs

    sibs = [s.item() for s in tree.iter_siblings(node_idx=node_idx, left=False)]
    assert [] == sibs
    assert tree.siblings(node_idx=node_idx, left=False).tolist() == sibs

    sibs = [s.item() for s in tree.iter_siblings(node_idx=node_idx, right=False)]
    assert [] == sibs
    assert tree.siblings(node_idx=node_idx, right=False).tolist() == sibs
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

    tree = new_tree(parents=parents)
    tree.pprint()

    node_idx = 0
    assert not tree.has_sibling(node_idx=node_idx)
    assert not tree.has_sibling(node_idx=node_idx, right=False)
    assert not tree.has_sibling(node_idx=node_idx, left=False)

    node_idx = 1
    assert tree.has_sibling(node_idx=node_idx)
    assert not tree.has_sibling(node_idx=node_idx, right=False)
    assert tree.has_sibling(node_idx=node_idx, left=False)

    node_idx = 2
    assert tree.has_sibling(node_idx=node_idx)
    assert not tree.has_sibling(node_idx=node_idx, right=False)
    assert tree.has_sibling(node_idx=node_idx, left=False)

    node_idx = 3
    assert tree.has_sibling(node_idx=node_idx)
    assert tree.has_sibling(node_idx=node_idx, right=False)
    assert tree.has_sibling(node_idx=node_idx, left=False)

    node_idx = 4
    assert tree.has_sibling(node_idx=node_idx)
    assert tree.has_sibling(node_idx=node_idx, right=False)
    assert tree.has_sibling(node_idx=node_idx, left=False)

    node_idx = 5
    assert tree.has_sibling(node_idx=node_idx)
    assert tree.has_sibling(node_idx=node_idx, right=False)
    assert tree.has_sibling(node_idx=node_idx, left=False)

    node_idx = 6
    assert tree.has_sibling(node_idx=node_idx)
    assert tree.has_sibling(node_idx=node_idx, right=False)
    assert not tree.has_sibling(node_idx=node_idx, left=False)

    node_idx = 7
    assert tree.has_sibling(node_idx=node_idx)
    assert tree.has_sibling(node_idx=node_idx, right=False)
    assert not tree.has_sibling(node_idx=node_idx, left=False)

    node_idx = 8
    assert not tree.has_sibling(node_idx=node_idx)
    assert not tree.has_sibling(node_idx=node_idx, right=False)
    assert not tree.has_sibling(node_idx=node_idx, left=False)

    node_idx = 9
    assert tree.has_sibling(node_idx=node_idx)
    assert not tree.has_sibling(node_idx=node_idx, right=False)
    assert tree.has_sibling(node_idx=node_idx, left=False)

    node_idx = 10
    assert tree.has_sibling(node_idx=node_idx)
    assert tree.has_sibling(node_idx=node_idx, right=False)
    assert tree.has_sibling(node_idx=node_idx, left=False)

    node_idx = 11
    assert tree.has_sibling(node_idx=node_idx)
    assert tree.has_sibling(node_idx=node_idx, right=False)
    assert not tree.has_sibling(node_idx=node_idx, left=False)

    with pytest.raises(IndexError):
        tree.has_sibling(node_idx=12)

    with pytest.raises(IndexError):
        tree.has_sibling(node_idx=12, left=False)

    with pytest.raises(IndexError):
        tree.has_sibling(node_idx=12, right=False)


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

    tree = new_tree(parents=parents)
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

    tree = new_tree(parents=parents)
    tree.pprint()

    node_idx = 0
    assert not tree.is_leaf(node_idx=node_idx)

    node_idx = 1
    assert not tree.is_leaf(node_idx=node_idx)

    node_idx = 2
    assert tree.is_leaf(node_idx=node_idx)

    node_idx = 3
    assert tree.is_leaf(node_idx=node_idx)

    node_idx = 4
    assert tree.is_leaf(node_idx=node_idx)

    node_idx = 5
    assert tree.is_leaf(node_idx=node_idx)

    node_idx = 6
    assert tree.is_leaf(node_idx=node_idx)

    node_idx = 7
    assert not tree.is_leaf(node_idx=node_idx)

    node_idx = 8
    assert not tree.is_leaf(node_idx=node_idx)

    node_idx = 9
    assert tree.is_leaf(node_idx=node_idx)

    node_idx = 10
    assert tree.is_leaf(node_idx=node_idx)

    node_idx = 11
    assert tree.is_leaf(node_idx=node_idx)

    with pytest.raises(IndexError):
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

    tree = new_tree(parents=parents)
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

    with pytest.raises(IndexError):
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

    tree = new_tree(parents=parents)
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

    tree = new_tree(parents=parents)
    tree.pprint()

    assert tree.parent(node_idx=0) is None
    for n_idx in range(1, len(parents)):
        assert tree.parent(node_idx=n_idx) == parents[n_idx]

    with pytest.raises(IndexError):
        tree.parent(node_idx=12)


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
    parents = [-1, 0, 1, 1, 1, 1, 1, 0, 7, 8, 8, 8]

    tree = new_tree(parents=parents)
    tree.pprint()
    for i in range(len(tree)):
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
    tree = new_tree(labels=tokens, parents=parents)
    tree.pprint()
    assert len(tree) == 12

    tree.delete_node(1, replacement_token="Qwertz")
    tree.pprint()
    assert len(tree) == 7


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
    tree = new_tree(labels=tokens, parents=parents)

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
    assert subtree.parent(7) is None

    return






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

    tree = new_tree(parents=parents)
    tree.pprint()

    new_tree = tree.delete_node(1)
    new_tree.pprint()
    assert len(new_tree) == len(tree) - 6
    assert len(tree) == len(parents), "original tree should stay the same"

    new_tree = tree.delete_node(7)
    new_tree.pprint()
    assert len(new_tree) == len(tree) - 7

    new_tree = tree.delete_node(14)
    new_tree.pprint()
    assert len(new_tree) == len(tree) - 2

    new_tree = tree.delete_node(4)
    new_tree.pprint()
    assert len(new_tree) == len(tree) - 1

    # delete root returns empty tree
    new_tree = tree.delete_node(0)
    assert len(new_tree) == 0
    print(new_tree)
    new_tree.pprint()


def test_tensortree_dunder_delete():
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

    tree = new_tree(parents=parents)
    tree.pprint()

    del tree[1]
    tree.pprint()
    assert len(tree) == len(parents) - 6
    return
    new_tree = tree.delete_node(7)
    new_tree.pprint()
    assert len(new_tree) == len(tree) - 7

    new_tree = tree.delete_node(14)
    new_tree.pprint()
    assert len(new_tree) == len(tree) - 2

    new_tree = tree.delete_node(4)
    new_tree.pprint()
    assert len(new_tree) == len(tree) - 1

    # delete root returns empty tree
    new_tree = tree.delete_node(0)
    assert len(new_tree) == 0
    print(new_tree)
    new_tree.pprint()


def test_tensortree_dunder_setitem():
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

    tree = new_tree(parents=parents)
    tree.pprint()

    tree[1], tree[7] = tree[7], tree[1]
    tree.pprint()
    return


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

    tree = new_tree(parents=parents)
    tree.pprint()

    new_tree = tree.delete_node(1, replacement_token=999)
    new_tree.pprint()
    assert len(new_tree) == len(tree) - 6 + 1
    assert len(tree) == len(parents), "original tree should stay the same"

    new_tree = tree.delete_node(7, replacement_token=999)
    new_tree.pprint()
    assert len(new_tree) == len(tree) - 7 + 1

    new_tree = tree.delete_node(14, replacement_token=999)
    new_tree.pprint()
    assert len(new_tree) == len(tree) - 2 + 1

    new_tree = tree.delete_node(4, replacement_token=999)
    new_tree.pprint()
    assert len(new_tree) == len(tree) - 1 + 1

    # delete root returns empty tree
    new_tree = tree.delete_node(0, replacement_token=999)
    assert len(new_tree) == 0 + 1
    new_tree.pprint()


def test_tensortree_replace_children_of_node():
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
    parents = [-1, 0, 1, 1, 1, 1, 1, 0, 7, 8]  # , 8, 8]

    tree = new_tree(parents=parents)
    tree.pprint()

    # tree.delete_children(1, replacement_token=999)
    # assert len(tree) == 8
    # tree.pprint()

    tree.replace_all_children_of_node(8, replacement_token=999)
    tree.pprint()


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

    tree = new_tree(parents=parents)
    tree.pprint()

    tree.swap(1, 8)
    tree.pprint()


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
    tree = new_tree(parents=parents)
    tree.pprint()

    merge_order = [4, 2, 1, 1, 1, 1, 1, 3, 2, 1, 1, 1]


    d = tree.descendants_for_subtree
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
