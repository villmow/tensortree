import unittest

from torchtree.data import sst
from torchtree.data.vocabulary import Vocabulary

import torchtree
import torch
import torch.nn.functional as F



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
    tree.pprint()
    return tree


def get_tree2():
    parents = [-1, 0, 1, 1, 3, 3, 0, 6, 7, 7, 6, 10, 10]

    # tokens = ["the cute dog is wagging its tail", "the cute dog", "the", "cute dog", "cute", "dog", "is wagging its tail", "is wagging", "is", "wagging", "its tail", "its", "tail"]
    # sample_tree = torchtree.new_tree(labels=tokens, parents=parents)
    tree = torchtree.new_tree(parents=parents, labels=list(range(19,32)))
    tree.pprint()
    return tree


def get_batched_tree(token_pad_idx, left_pad=False):
    tree1 = get_tree()
    tree2 = get_tree2()

    batched_tokens = torchtree.utils.collate_tokens(
        [tree1.data.node_data, tree2.data.node_data],
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


class TestTreeLSTM(unittest.TestCase):

    def test_sum_child_features(self):
        tokens, descendants, parents = get_batched_tree(token_pad_idx=-1, left_pad=True)

        # we need scale parents, so that every sample in batch gets different parents
        # the actual parent value is not needed


        B, N = tokens.shape
        D = 2  # Embedding dim,

        scale = torch.arange(0, B).view(-1, 1) * N
        # print("scale", scale)
        adjusted_parents = parents + scale

        print(parents)
        print(adjusted_parents)

        node_indicences = torchtree.node_incidence_matrix(descendants, pad_idx=0)

        leaves = node_indicences.sum(-2) == 1
        print(leaves)
        print("node_indicences.shape", node_indicences.shape)
        # print(node_indicences.int())
        print(descendants.int())

        # # h and c states for every node in the batch
        # h = descendants.new_zeros(B, N, D, dtype=torch.float)
        h = torch.arange(N*B).view(B, N, 1).repeat(1, 1, D).float()
        print("h.shape", h.shape)
        print("h", h)
        features = torch.zeros_like(h, dtype=torch.float)

        nodes_in_previous_layer = None
        for i, (level, nodes_in_layer) in enumerate(torchtree.mask_level(node_indicences, return_level=True)):
            print("#" * 50)
            print(level)
            print("nodes_in_previous_layer", nodes_in_previous_layer)
            print("nodes_in_layer", nodes_in_layer)
            print("nodes_in_layer.shape", nodes_in_layer.shape)

            if i > 0:
                print("-" * 20)
                print("parents[nodes_in_previous_layer]", parents[nodes_in_previous_layer])
                print("adjusted_parents[nodes_in_previous_layer]", adjusted_parents[nodes_in_previous_layer])
                # print(parents[nodes_in_layer.nonzero(as_tuple=True)])

                _, num_children = adjusted_parents[nodes_in_previous_layer].unique_consecutive(return_counts=True)

                print("num_children", num_children)
                print("num_children.cumsum(-1)")
                offsets = num_children.cumsum(-1)
                offsets = torch.cat(
                    [offsets.new_zeros(1), offsets[:-1]]
                )
                print("num_children.shape", num_children.shape)
                print("offsets", offsets)
                print("offsets.shape", offsets.shape)
                # num_children = torch.roll(num_children, 1)  # FIXME why?
                # print("num_children rolled", num_children)
                # print("num_children rolled.shape", num_children.shape)

                # offsets = num_children.cumsum(-1) - 1  # [ Total nodes in layers ]
                # print("offsets", offsets)
                # print("offsets.shape", offsets.shape)
                child_features = h[nodes_in_previous_layer]  # [ Total nodes in layers ]
                print("child_features", child_features)
                print("child_features.shape", child_features.shape)
                parent_features = F.embedding_bag(
                    child_features.float(),
                    torch.arange(child_features.size(0)),
                    offsets,
                    mode="sum"
                )

                # we did not compute anything for leaves yet
                nodes_to_merge = (~leaves) & nodes_in_layer

                print("tokens[nodes_to_merge]", tokens[nodes_to_merge])
                features[nodes_to_merge] = parent_features

                print('~' * 10)

            leaves_to_merge = leaves & nodes_in_layer
            features[leaves_to_merge] = h[leaves_to_merge]
            nodes_in_previous_layer = nodes_in_layer

            print(features)

    def test_sum_child_features_mat(self):
        tokens, descendants, parents = get_batched_tree(token_pad_idx=-1, left_pad=False)

        # we need scale parents, so that every sample in batch gets different parents
        # the actual parent value is not needed
        pad_mask = descendants == 0
        print("pad_mask", pad_mask)
        print("pad_mask.shape", pad_mask.shape)
        B, N = tokens.shape
        D = 2  # Embedding dim,

        scale = torch.arange(0, B).view(-1, 1) * N
        # print("scale", scale)
        adjusted_parents = parents + scale

        print(parents)
        print(adjusted_parents)
        for paremt in parents:
            print("..."*30)
            p = paremt[paremt != -2] #- (paremt == -2).sum(-1)
            print("p", p.int())
            adj = torchtree.adjacency_matrix_unbatched(p, directed=True)
            print("adj", adj.int())
            print("adj", adj.int().sum(-1))

        print("##"*30)
        adj = torchtree.adjacency_matrix(parents, directed=True)
        print("adj", adj.int())

        node_indicences = torchtree.node_incidence_matrix(descendants, pad_idx=0)
        print(node_indicences.int())
        leaves = node_indicences.sum(-2) == 1
        print(leaves)
        print("node_indicences.shape", node_indicences.shape)
        # print(node_indicences.int())
        print(descendants.int())

        # # h and c states for every node in the batch
        # h = descendants.new_zeros(B, N, D, dtype=torch.float)
        h = torch.arange(N*B).view(B, N, 1).repeat(1, 1, D).float()
        print("h.shape", h.shape)
        print("h", h)
        features = torch.zeros_like(h, dtype=torch.float)

        nodes_in_previous_layer = None
        for i, (level, nodes_in_layer) in enumerate(torchtree.mask_level(node_indicences, return_level=True)):
            print("#" * 50)
            print(level)
            print("nodes_in_previous_layer", nodes_in_previous_layer)
            print("nodes_in_layer", nodes_in_layer)
            print("nodes_in_layer.shape", nodes_in_layer.shape)

            if i > 0:
                print("-" * 20)
                print("parents[nodes_in_previous_layer]", parents[nodes_in_previous_layer])
                print("adjusted_parents[nodes_in_previous_layer]", adjusted_parents[nodes_in_previous_layer])
                # print(parents[nodes_in_layer.nonzero(as_tuple=True)])

                _, num_children = adjusted_parents[nodes_in_previous_layer].unique_consecutive(return_counts=True)

                print("num_children", num_children)
                print("num_children.cumsum(-1)")
                offsets = num_children.cumsum(-1)
                offsets = torch.cat(
                    [offsets.new_zeros(1), offsets[:-1]]
                )
                print("num_children.shape", num_children.shape)
                print("offsets", offsets)
                print("offsets.shape", offsets.shape)
                # num_children = torch.roll(num_children, 1)  # FIXME why?
                # print("num_children rolled", num_children)
                # print("num_children rolled.shape", num_children.shape)

                # offsets = num_children.cumsum(-1) - 1  # [ Total nodes in layers ]
                # print("offsets", offsets)
                # print("offsets.shape", offsets.shape)
                child_features = h[nodes_in_previous_layer]  # [ Total nodes in layers ]
                print("child_features", child_features)
                print("child_features.shape", child_features.shape)
                parent_features = F.embedding_bag(
                    child_features.float(),
                    torch.arange(child_features.size(0)),
                    offsets,
                    mode="sum"
                )

                # we did not compute anything for leaves yet
                nodes_to_merge = (~leaves) & nodes_in_layer

                print("tokens[nodes_to_merge]", tokens[nodes_to_merge])
                features[nodes_to_merge] = parent_features

                print('~' * 10)

            leaves_to_merge = leaves & nodes_in_layer
            features[leaves_to_merge] = h[leaves_to_merge]
            nodes_in_previous_layer = nodes_in_layer

            print(features)

    def test_index_add(self):
        tokens, descendants, parents = get_batched_tree(token_pad_idx=-1, left_pad=False)

        # we need scale parents, so that every sample in batch gets different parents
        # the actual parent value is not needed
        pad_mask = descendants == 0
        print("pad_mask", pad_mask)
        print("pad_mask.shape", pad_mask.shape)
        B, N = tokens.shape
        D = 2  # Embedding dim,

        scale = torch.arange(0, B).view(-1, 1) * N
        # print("scale", scale)
        adjusted_parents = parents + scale

        print(parents)
        print(adjusted_parents)
        for paremt in parents:
            print("..." * 30)
            p = paremt[paremt != -2]  # - (paremt == -2).sum(-1)
            print("p", p.int())
            adj = torchtree.adjacency_matrix_unbatched(p, directed=True)
            print("adj", adj.int())
            print("adj", adj.int().sum(-1))

        print("##" * 30)
        adj = torchtree.adjacency_matrix(parents, directed=True)
        print("adj", adj.int())

        node_indicences = torchtree.node_incidence_matrix(descendants, pad_idx=0)
        print(node_indicences.int())
        leaves = node_indicences.sum(-2) == 1
        print(leaves)
        print("node_indicences.shape", node_indicences.shape)
        # print(node_indicences.int())
        print(descendants.int())

        # # h and c states for every node in the batch
        # h = descendants.new_zeros(B, N, D, dtype=torch.float)
        h = torch.arange(N * B).view(B, N, 1).repeat(1, 1, D).float()
        print("h.shape", h.shape)
        print("h", h)
        nodes = torch.ones(B, N, D, dtype=torch.float)
        previous = None
        for iteration, (level, nodes_in_layer) in enumerate(
                torchtree.mask_level(node_indicences, return_level=True)):
            nodes.view(-1, D).index_add_(0, adjusted_parents[nodes_in_layer], nodes[nodes_in_layer])
            print(nodes_in_layer)
            print(iteration)
            print("nodes[nodes_in_layer]", nodes[nodes_in_layer])
            print("nodes[nodes_in_layer].shape", nodes[nodes_in_layer].shape)
            print("adjusted_parents[nodes_in_layer]", adjusted_parents[nodes_in_layer])
            print("adjusted_parents[nodes_in_layer].shape", adjusted_parents[nodes_in_layer].shape)

            if iteration > 0:
                print("nodes[previous]", nodes[previous])
                print("nodes[previous].shape", nodes[previous].shape)

                nodes.index_add_(0, adjusted_parents[nodes_in_layer], nodes[nodes_in_layer])

            print(nodes)
            previous = nodes_in_layer