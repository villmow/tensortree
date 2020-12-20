from typing import *

import hydra
import pytorch_lightning as pl
import torch

import torchtree
from torchtree.data.vocabulary import Vocabulary
from treemodels.config import OptimizerConfig


class TreeLSTMClassification(pl.LightningModule):
    def __init__(
        self,
        cfg: OptimizerConfig,
        lr = 0.001,
        weight_decay = 1e-4,
        embedding_dim = 150,
        embedding_file: Optional[str] = None,
        tree_lstm_hidden_size = 150,
        dropout = 0.1,
        num_classes=5,
        vocabulary: Vocabulary = None,
        **kwargs
    ):
        super().__init__()

        self.save_hyperparameters()

        self.vocabulary = vocabulary
        self.optimizer_cfg = cfg

        self.embedding = torch.nn.Embedding(
            len(self.vocabulary), embedding_dim,
            padding_idx=self.vocabulary.pad_index
        )
        if embedding_file is not None:
            embed_dict = torchtree.utils.parse_embedding(embedding_file)
            torchtree.utils.print_embed_overlap(embed_dict, self.vocabulary)
            torchtree.utils.load_embedding(embed_dict, self.vocabulary, self.embedding)

        self.lstm = torchtree.models.TreeLSTM(embedding_dim, tree_lstm_hidden_size)
        self.classifier = torch.nn.Linear(tree_lstm_hidden_size, num_classes)
        self.loss = torch.nn.CrossEntropyLoss(reduction="mean")
        self.dropout = torch.nn.Dropout(p=dropout)

        self.train_acc = pl.metrics.Accuracy()
        self.train_root_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()
        self.valid_root_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()
        self.test_root_acc = pl.metrics.Accuracy()

    def print_batch(self, batch):
        from torchtree import new_tree

        tree = batch["tree"]
        print("Trees in batch:", tree["tokens"].size(0))
        for tokens, descendants, parents in zip(tree["tokens"], tree["descendants"], tree["parents"]):
            num_pad = (tokens == self.vocabulary.pad_index).sum()
            tree = new_tree(
                labels=self.vocabulary.decode_tokens(tokens)[num_pad:],
                descendants=(descendants - 1)[num_pad:],
                parents=parents[parents > -2] - (parents <= -2).sum()
            )
            tree.pprint()

    def forward(self, tree: Dict[str, torch.Tensor]) -> torch.Tensor:
        tokens, descendants, parents = tree["tokens"], tree["descendants"], tree["parents"]
        # embed all inputs
        embed = self.embedding(tokens)
        embed = self.dropout(embed)

        # feed through tree_lstm
        hidden, memory = self.lstm(embed, descendants, parents)
        hidden = self.dropout(hidden)

        # and classify
        logits = self.classifier(hidden)

        return logits

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        # self.print_batch(batch)

        tree = batch["tree"]
        targets = batch.pop("labels")

        logits = self(tree)  # [B, N, D]

        # compute loss only on actual tokens, so either
        # - filter padding tokens/logits (done here)
        # - or assign a unique ignored label to padding tokens (not done)
        token_mask = tree["tokens"] != self.vocabulary.pad_index  # [B, N]

        y_hat = logits[token_mask]  # [S, D] (S <= B*N)
        y = targets[token_mask]     # [S]

        loss = self.loss(y_hat, y)
        self.train_acc(y_hat, y)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=False, prog_bar=True)

        self.train_root_acc(logits[:,0,:], targets[:,0])
        self.log('train_root_acc', self.train_root_acc, on_step=True, on_epoch=False, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        tree = batch["tree"]
        targets = batch.pop("labels")

        logits = self(tree)  # [B, N, D]

        # compute loss only on actual tokens, so either
        # - filter padding tokens/logits (done here)
        # - or assign a unique ignored label to padding tokens (not done)
        token_mask = tree["tokens"] != self.vocabulary.pad_index  # [B, N]

        y_hat = logits[token_mask]
        y = targets[token_mask]

        loss = self.loss(y_hat, y)

        self.valid_acc(y_hat, y)
        self.log('valid_acc', self.valid_acc, on_step=False, on_epoch=True, prog_bar=True)

        self.valid_root_acc(logits[:,0,:], targets[:,0])
        self.log('valid_root_acc', self.valid_root_acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        # self.print_batch(batch)

        tree = batch["tree"]
        targets = batch.pop("labels")

        logits = self(tree)  # [B, N, D]

        # compute loss only on actual tokens, so either
        # - filter padding tokens/logits (done here)
        # - or assign a unique ignored label to padding tokens (not done)
        token_mask = tree["tokens"] != self.vocabulary.pad_index  # [B, N]

        y_hat = logits[token_mask]
        y = targets[token_mask]

        loss = self.loss(y_hat, y)

        self.test_acc(y_hat, y)
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

        self.test_root_acc(logits[:, 0, :], targets[:, 0])
        self.log('test_root_acc', self.test_root_acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        no_decay = []
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
        ]

        optimizer = hydra.utils.instantiate(
            self.optimizer_cfg,
            optimizer_grouped_parameters,
        )
        print(optimizer)
        return optimizer

