from dataclasses import dataclass
from typing import Optional

from omegaconf import MISSING

from torchtree.utils import get_project_root

@dataclass
class DatasetConfig:
    data_dir: str = str(get_project_root() / "data")
    train_batch_size: int = 25
    eval_batch_size: int = 25
    force_reload: bool = False


@dataclass
class SSTConfig(DatasetConfig):
    _target_: str = "torchtree.data.sst.SSTDatamodule"
    data_dir: str = str(get_project_root() / "data/sst")


@dataclass
class TreeLSTMConfig:
    _target_: str = "treemodels.treelstm.TreeLSTMClassification"
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    embedding_dim: int = 150
    tree_lstm_hidden_size: int = 150
    dropout: float = 0.1


@dataclass
class TrainConfig:
    dataset: DatasetConfig = SSTConfig()
    model: TreeLSTMConfig = TreeLSTMConfig()

    min_epochs: int = 1
    max_epochs: Optional[int] = None

    wandb_offline: bool = True
    seed: int = 1234


from hydra.core.config_store import ConfigStore

cs = ConfigStore.instance()
cs.store(name="config", node=TrainConfig)

cs.store(group="model", name="treelstm", node=TreeLSTMConfig)
cs.store(group="data", name="sst", node=SSTConfig)
