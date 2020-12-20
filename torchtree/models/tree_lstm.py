import torch
import torch.nn.functional as F

import torchtree


class TreeLSTM(torch.nn.Module):
    """PyTorch TreeLSTM model that implements efficient batching."""

    def __init__(self, in_features, out_features):
        """TreeLSTM class initializer
        Takes in int sizes of in_features and out_features and sets up model Linear network layers.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # bias terms are only on the W layers for efficiency
        self.W_iou = torch.nn.Linear(self.in_features, 3 * self.out_features)
        self.U_iou = torch.nn.Linear(self.out_features, 3 * self.out_features, bias=False)

        # f terms are maintained seperate from the iou terms because they involve sums over child nodes
        # while the iou terms do not
        self.W_f = torch.nn.Linear(self.in_features, self.out_features)
        self.U_f = torch.nn.Linear(self.out_features, self.out_features, bias=False)

    def forward(self, features, descendants, parents):
        """Run TreeLSTM model on a tree data structure with node features
        Takes Tensors encoding node features, a tree node adjacency_list, and the order in which
        the tree processing should proceed in node_order and edge_order.
        """
        B, N, D = features.shape

        # binary incidence matrix
        node_indicences = torchtree.node_incidence_matrix(descendants, pad_idx=0)

        # mask over all leaves
        leaves = node_indicences.sum(-2) == 1

        # adjust parents, so that every sample in batch gets unique parents
        # the actual parent index is not needed.
        scale = torch.arange(0, B, device=parents.device).view(-1, 1) * N
        adjusted_parents = parents + scale

        # # h and c states for every node in the batch
        h = descendants.new_zeros(B, N, self.out_features, dtype=torch.float)
        c = descendants.new_zeros(B, N, self.out_features, dtype=torch.float)

        nodes_in_previous_layer = None  # mask of nodes in previous iteration
        for level, nodes_in_layer in torchtree.mask_level(node_indicences, return_level=True):
            # print("##"* 40)
            # print("at level", level)

            # mask over all leaves that we handle in this iteration
            leaves_in_layer = leaves & nodes_in_layer
            # leaves_in_layer = leaves_in_layer[:, :, None]  # unsqueeze

            # mask over all nodes that we handle in this iteration
            nonterminals_in_layer = (~leaves) & nodes_in_layer
            # nonterminals_in_layer = nonterminals_in_layer[:, :, None]  # unsqueeze

            # print("nodes_in_layer", nodes_in_layer.sum())
            # print("leaves_in_layer", leaves_in_layer.sum())
            # print("nonterminals_in_layer", nonterminals_in_layer.sum())

            # x = features.masked_select(nodes_in_layer).view(-1, H)  # [Total nodes in all trees, H]
            x = features[nodes_in_layer]
            # print("x.shape", x.shape)

            x_nonterminal_mask = nonterminals_in_layer[nodes_in_layer]  # [B]
            # print("x_nonterminal_mask.shape", x_nonterminal_mask.shape)

            # input all node features
            iou = self.W_iou(x)

            # now we need to merge children for all nonterminals
            if nodes_in_previous_layer is not None:
                # count the amount of children for every nonterminal
                _, num_children = adjusted_parents[nodes_in_previous_layer].unique_consecutive(return_counts=True)

                assert num_children.size(0) == x_nonterminal_mask.sum()

                # compute offsets for embedding bag
                offsets = num_children.cumsum(-1)
                offsets = torch.cat(
                    [offsets.new_zeros(1), offsets[:-1]]
                )

                child_h = h[nodes_in_previous_layer]
                child_idx = torch.arange(child_h.size(0), device=child_h.device)

                parent_features = F.embedding_bag(
                    child_idx,
                    child_h,
                    offsets,
                    mode="sum"
                )

                iou[x_nonterminal_mask, :] += self.U_iou(parent_features)

            # print("iou.shape", iou.shape)
            # equation (1), (3), (4)
            i, o, u = torch.chunk(iou, 3, 1)
            # print("i.shape", i.shape)
            # print("o.shape", o.shape)
            # print("u.shape", u.shape)

            i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)

            c[nodes_in_layer] = i * u

            if nodes_in_previous_layer is not None:
                f = self.W_f(features[nonterminals_in_layer])
                # print("f.shape", f.shape)

                f = torch.repeat_interleave(f, num_children, dim=0)
                # print("f.shape", f.shape)

                f += self.U_f(child_h)
                # print("f.shape", f.shape)

                fc = f * c[nodes_in_previous_layer]
                child_sum = F.embedding_bag(
                    child_idx,
                    fc,
                    offsets,
                    mode="sum"
                )

                c[nonterminals_in_layer] += child_sum

            # equation (6)
            h[nodes_in_layer] = o * torch.tanh(c[nodes_in_layer])
            nodes_in_previous_layer = nodes_in_layer

        return h, c
