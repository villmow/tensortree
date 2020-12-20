# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Stanford Sentiment Treebank dataset.

Each sample is the constituency tree of a sentence. The leaf nodes
represent words. Non-leaf nodes have a special value ``<NT>``.

The tree data is stored in a parent pointer format and can be loaded with
TreeStorage or TensorTree (use new_tree constructor) .

Each node has a sentiment annotation: 5 classes (very negative,
negative, neutral, positive and very positive).

Statistics:

- Train examples: 8,544
- Dev examples: 1,101
- Test examples: 2,210
- Number of classes for each node: 5

"""
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Tuple, List, Union

import datasets
from nltk.corpus.reader import BracketParseCorpusReader
from nltk.tree import Tree
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from torchtree import TreeStorage
from torchtree.data.vocabulary import Vocabulary
from torchtree.utils import collate_tree


_CITATION = """
@inproceedings{socher2013recursive,
  title={Recursive deep models for semantic compositionality over a sentiment treebank},
  author={Socher, Richard and Perelygin, Alex and Wu, Jean and Chuang, Jason and Manning, Christopher D and Ng, Andrew Y and Potts, Christopher},
  booktitle={Proceedings of the 2013 conference on empirical methods in natural language processing},
  pages={1631--1642},
  year={2013}
}
"""

_DESCRIPTION = """
Semantic word spaces have been very useful but cannot express the meaning of longer phrases in a principled way. Further progress towards understanding compositionality in tasks such as sentiment detection requires richer supervised training and evaluation resources and more powerful models of composition. To remedy this, we introduce a Sentiment Treebank. It includes fine grained sentiment labels for 215,154 phrases in the parse trees of 11,855 sentences and presents new challenges for sentiment compositionality.
"""

_HOMEPAGE = "http://nlp.stanford.edu/sentiment/index.html"

_LICENSE = ""

_URLs = {
    'fine_grained': "https://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip",
}


class SST(datasets.GeneratorBasedBuilder):
    """Stanford Sentiment Treebank dataset.."""

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="fine_grained", version=VERSION, description=""),
    ]

    def _info(self):
        features = datasets.Features(
            {
                "tree": {
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "parents": datasets.Sequence(datasets.Value("int16")),
                    "descendants": datasets.Sequence(datasets.Value("int16"))
                },
                "labels": datasets.Sequence(
                    datasets.features.ClassLabel(
                        num_classes=5,
                        names=[
                            "very negative",
                            "negative",
                            "neutral",
                            "positive",
                            "very positive"
                        ]
                    )
                )
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        my_urls = _URLs[self.config.name]
        data_dir = dl_manager.download_and_extract(my_urls)
        data_dir = Path(data_dir).absolute()
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_dir / "trees/train.txt",
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": data_dir / "trees/test.txt",
                    "split": "test"
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": data_dir / "trees/dev.txt",
                    "split": "valid",
                },
            ),
        ]

    def _generate_examples(self, filepath, split):
        """ Yields examples. """
        corpus = BracketParseCorpusReader(
            str(filepath.parent), [str(filepath)]
        )
        nltk_trees = corpus.parsed_sents(str(filepath))

        for i, nltk_tree in enumerate(nltk_trees):
            tree_storage, labels = build_tree(nltk_tree)
            yield i, {
                "tree": {
                    "tokens": tree_storage.labels,
                    "parents": tree_storage.parents,
                    "descendants": tree_storage.descendants,
                },
                "labels": labels,
            }


def build_tree(root: Tree) -> Tuple[TreeStorage, List[int]]:
    labels = []
    parents = []
    targets = []

    def from_nltk(tree: Union[str, Tree], parent=-1):
        """

        :param tree: a NLTK Tree object or a string
        :param parent: a parent astlib Tree
        :return:
        """

        is_leaf = isinstance(tree[0], (str, bytes))

        word = tree[0].lower() if is_leaf else "<NT>"
        score = int(tree.label())

        new_node_idx = len(parents)
        labels.append(word)
        parents.append(parent)
        targets.append(score)

        if not is_leaf:
            for child in tree:
                from_nltk(child, parent=new_node_idx)

    from_nltk(root)
    assert len(labels) == len(targets) == len(parents)

    return TreeStorage(parents=parents, labels=labels), torch.tensor(targets, dtype=torch.uint8)


def build_vocab():
    sst_dataset = datasets.load_dataset(__file__)

    # we build vocab on all splits (to use pretrained word embeddings later on)
    # all_splits = datasets.concatenate_datasets(list(sst_dataset.values()))
    all_splits = sst_dataset["train"]

    vocab = Vocabulary()
    for tree in all_splits["tree"]:
        for token in tree["tokens"]:
            vocab.add_symbol(token)
    vocab.finalize()
    return vocab


def prepare(directory: str):
    """ Binarizes and saves a dataset along with its vocab to disk """
    vocab = build_vocab()
    outdir = Path(directory)

    def binarize(example):
        example["tree"]["tokens"] = vocab.encode_tokens(example["tree"]["tokens"])
        return example

    dataset = datasets.load_dataset(__file__, cache_dir=outdir)
    dataset = dataset.map(binarize, batched=False)
    dataset.set_format(type="torch")

    outdir.mkdir(exist_ok=True)
    vocab.save(str((outdir / "vocab.txt").absolute()))
    dataset.save_to_disk(outdir)


class SSTDatamodule(pl.LightningDataModule):
    """ The SST dataset binarized and ready to use. """
    def __init__(
            self, data_dir, train_batch_size, eval_batch_size, force_reload,
            train_dataloader_worker, eval_dataloader_worker
    ):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.data_dir = Path(data_dir)
        self.force_reload = force_reload
        self.train_dataloader_worker = train_dataloader_worker
        self.eval_dataloader_worker = eval_dataloader_worker


    def prepare_data(self):
        """ Executed once """
        if not self.data_dir.exists() or self.force_reload:
            prepare(self.data_dir)
            self.force_reload = False

    def setup(self, stage=None):
        # load datasets
        self.dataset = datasets.load_from_disk(self.data_dir)

        try:
            self.vocab = Vocabulary.load(self.data_dir / "vocab.txt")
            self.collater = partial(collate_tree, pad_idx=self.vocab.pad_index)

        except FileNotFoundError:
            raise FileNotFoundError("Vocab not found. Please execute prepare before.")

    def train_dataloader(self):
        return DataLoader(
            self.dataset['train'], batch_size=self.train_batch_size, collate_fn=self.collater,
            num_workers=self.train_dataloader_worker
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset['validation'], batch_size=self.eval_batch_size, collate_fn=self.collater,
            num_workers=self.eval_dataloader_worker
        )

    def test_dataloader(self):
        return DataLoader(self.dataset['test'], batch_size=self.eval_batch_size, collate_fn=self.collater,
            num_workers=self.eval_dataloader_worker
      )

    @property
    def class_label(self) -> datasets.ClassLabel:
        if not hasattr(self, "dataset"):
            self.setup()

        return self.dataset["train"].features['labels'].feature
