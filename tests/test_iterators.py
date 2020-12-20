from unittest import TestCase

import torch
import torchtree


def get_tree():
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

    tree = torchtree.new_tree(parents=parents)
    return tree


def get_tree2():
    parents = [-1, 0, 1, 1, 3, 3, 0, 6, 7, 7, 6, 10, 10]
    # tokens = ["the cute dog is wagging its tail", "the cute dog", "the", "cute dog", "cute", "dog", "is wagging its tail", "is wagging", "is", "wagging", "its tail", "its", "tail"]
    # sample_tree = torchtree.new_tree(labels=tokens, parents=parents)
    sample_tree = torchtree.new_tree(parents=parents)

    return sample_tree


def get_batched_tree(token_pad_idx, left_pad=False):
    tree1 = get_tree()
    tree2 = get_tree2()

    batched_tokens = torchtree.utils.collate_tokens(
        [tree1.data.labels, tree2.data.labels],
        pad_idx=token_pad_idx,
        left_pad=left_pad
    )
    batched_descendants = torchtree.utils.collate_descendants(
        [tree1.data.descendants, tree2.data.descendants],
        left_pad=left_pad
    )
    batched_parents = torchtree.utils.collate_parents(
        [tree1.data.parents, tree2.data.parents],
        left_pad=left_pad
    )
    return batched_tokens, batched_descendants, batched_parents



class TestIterators(TestCase):
    def test_mask_level(self):
        tree = get_tree()

        ni = torchtree.node_incidence_matrix(tree.data.descendants)

        masks = list(torchtree.mask_level(ni))

        # check that we visited no node twice
        for level_mask in masks:
            for other_level_mask in masks:
                if level_mask is other_level_mask:
                    continue
                assert not (level_mask & other_level_mask).any(), "no node should be visited twice"

        # check that we visited all nodes
        check = ni.new_zeros(ni.shape[0], dtype=torch.bool)
        for level_mask in masks:
            check |= level_mask
        assert check.all(), "everything should be true after visiting all layers"

    def test_mask_level_batched(self):
        tree = get_tree()
        tree2 = get_tree2()
        tree.pprint()
        tree2.pprint()

        tokens, descendants, parents = get_batched_tree(token_pad_idx=-1, left_pad=True)
        pad_mask = tokens == -1

        ni = torchtree.node_incidence_matrix(descendants, pad_idx=0)

        for level, mask in torchtree.mask_level(ni, return_level=True):
            print(f"tokens in level {level}:", tokens[mask])

        masks = list(torchtree.mask_level(ni))

        # check that we visited no node twice
        for level_mask in masks:
            for other_level_mask in masks:
                if level_mask is other_level_mask:
                    continue
                assert not (level_mask & other_level_mask).any(), "no node should be visited twice"

        # check that we visited all nodes
        check = torch.zeros_like(tokens, dtype=torch.bool)
        for level_mask in masks:
            print(level_mask)
            check |= level_mask
        assert check[~pad_mask].all(), "everything should be true after visiting all layers"
        assert not check[pad_mask].all(), "padding tokens should be never visited"

    def test_mask_level_batched_right_padded(self):
        tree = get_tree()
        tree2 = get_tree2()
        tree.pprint()
        tree2.pprint()

        tokens, descendants, parents = get_batched_tree(token_pad_idx=-1, left_pad=False)
        pad_mask = tokens == -1

        print(tokens)
        ni = torchtree.node_incidence_matrix(descendants, pad_idx=0)

        for level, mask in torchtree.mask_level(ni, return_level=True):
            print(f"tokens in level {level}:", tokens[mask])

        masks = list(torchtree.mask_level(ni))

        # check that we visited no node twice
        for level_mask in masks:
            for other_level_mask in masks:
                if level_mask is other_level_mask:
                    continue
                assert not (level_mask & other_level_mask).any(), "no node should be visited twice"

        # check that we visited all nodes
        check = torch.zeros_like(tokens, dtype=torch.bool)
        for level_mask in masks:
            print(level_mask)
            check |= level_mask
        assert check[~pad_mask].all(), "everything should be true after visiting all layers"
        assert not check[pad_mask].all(), "padding tokens should be never visited"

    def test_batch_parents(self):
        tree = get_tree()
        tree2 = get_tree2()
        tree.pprint()
        tree2.pprint()

        tokens_right, descendants_right, parents_right = get_batched_tree(token_pad_idx=-1, left_pad=False)
        tokens_left, descendants_left, parents_left = get_batched_tree(token_pad_idx=-1, left_pad=False)

        ni_right = torchtree.node_incidence_matrix(descendants_right, pad_idx=0)
        ni_left = torchtree.node_incidence_matrix(descendants_left, pad_idx=0)

        for (level1, mask1, ), (level2, mask2,) in zip(
                torchtree.mask_level(ni_right, return_level=True),
                torchtree.mask_level(ni_left, return_level=True),
        ):
            assert level1 == level2
            print("#"*20)
            print(f"tokens in level {level1}:", tokens_right[mask1])
            print(f"tokens in level {level2}:", tokens_left[mask2])
            print(f"parents of tokens in level {level2}:", parents_right[mask1])
            print(f"parents of tokens in level {level2}:", parents_left[mask2])
            assert (tokens_right[mask1] == tokens_left[mask2]).all()
            assert (parents_left[mask2] == parents_right[mask1]).all()