import unittest

from torchtree.data.sst import SST
from torchtree.data.vocabulary import Vocabulary


class TestSST(unittest.TestCase):
    def test_process(self):
        train = SST(mode="train")
        assert train[0] is not None
        valid = SST(mode="dev")
        assert valid[0] is not None
        test = SST(mode="test")
        assert test[0] is not None

        train_vocab = train.build_vocab()
        valid_vocab = valid.build_vocab()
        test_vocab = test.build_vocab()

        train_vocab.save("sst-train.vocab")
        valid_vocab.save("sst-valid.vocab")
        test_vocab.save("sst-test.vocab")

        vocab = Vocabulary()
        vocab.update(train_vocab)
        vocab.update(valid_vocab)
        vocab.update(test_vocab)
        vocab.finalize()
        vocab.save("sst.vocab")