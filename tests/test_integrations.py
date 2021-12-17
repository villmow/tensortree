import pytest
import torch


import tensortree
from tensortree import TensorTree, TreeStorage
from tensortree import parents_from_descendants, descendants_from_parents


def test_from_spacy():
    import spacy

    nlp = spacy.load("en_core_web_sm")
    doc = nlp("Autonomous cars shift insurance liability toward manufacturers")

    from tensortree.integrations import from_spacy
    from pprint import pprint
    pprint(doc.to_json())

    sent = next(doc.sents)
    tree = from_spacy(sent)
    tree.pprint()

