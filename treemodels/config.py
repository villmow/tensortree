from dataclasses import dataclass, field
from typing import Optional, List, Any

from omegaconf import MISSING

from torchtree.utils import get_project_root

##########################################################
# datasets
##########################################################
@dataclass
class DatasetConfig:
    data_dir: str = str(get_project_root() / "data")
    force_reload: bool = False


@dataclass
class SSTConfig(DatasetConfig):
    _target_: str = "torchtree.data.sst.SSTDatamodule"
    data_dir: str = str(get_project_root() / "data/sst")


##########################################################
# optimizer
##########################################################
@dataclass
class OptimizerConfig:
    _target_: str = MISSING
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    no_decay_params: Optional[List[str]] = None


@dataclass
class AdamOptimizer(OptimizerConfig):
    _target_: str = "torch.optim.Adam"


@dataclass
class AdaGradOptimizer(OptimizerConfig):
    _target_: str = "torch.optim.AdaGrad"


##########################################################
# model
##########################################################
@dataclass
class ModelConfig:
    _target_: str = MISSING
    optimizer: OptimizerConfig = MISSING

@dataclass
class TreeLSTMConfig(ModelConfig):
    _target_: str = "treemodels.treelstm.TreeLSTMClassification"
    embedding_dim: int = 150
    tree_lstm_hidden_size: int = 150
    dropout: float = 0.1


##########################################################
# Logging
##########################################################
@dataclass
class LogConfig:
    pass


@dataclass
class Wandb(LogConfig):
    _target_: str = "pytorch_lightning.loggers.WandbLogger"
    offline: bool = True
    project: str = MISSING


##########################################################
# Training
##########################################################
@dataclass
class LightningTrainerConfig:
    # default_root_dir: Optional[str] = None
    gradient_clip_val: float = 0
    gpus: Optional[int] = None
    auto_select_gpus: bool = False
    log_gpu_memory: Optional[str] = None
    overfit_batches: float = 0.0
    track_grad_norm: int = -1
    accumulate_grad_batches: int = 1
    max_epochs: int = 1000
    min_epochs: int = 1
    max_steps: Optional[int] = None
    min_steps: Optional[int] = None
    limit_train_batches: float = 1.0
    limit_val_batches: float = 1.0
    limit_test_batches:  float = 1.0
    val_check_interval: float = 1.0
    precision: int = 32
    resume_from_checkpoint: Optional[str] = None
    deterministic: bool = False
    reload_dataloaders_every_epoch: bool = False
    auto_lr_find: bool = False
    auto_scale_batch_size: bool = False


defaults = [
    # Load the config "sst" from the config group "dataset"
    {"dataset": "sst"},
    {"model": "treelstm"},
    {"logger": "wandb"},
    {"pl_trainer": "default"},
]

@dataclass
class TrainConfig:
    # fill defaults from above
    defaults: List[Any] = field(default_factory=lambda: defaults)

    pl_trainer: LightningTrainerConfig = MISSING
    dataset: DatasetConfig = MISSING
    model: ModelConfig = MISSING
    logger: LogConfig = MISSING

    train_batch_size: int = 25
    eval_batch_size: int = 25

    seed: int = 1234



from hydra.core.config_store import ConfigStore

cs = ConfigStore.instance()
# cs.store(group="schema/model", name="treelstm", node=TreeLSTMConfig, package="model")
cs.store(group="dataset", name="sst", node=SSTConfig)

cs.store(group="model", name="treelstm", node=TreeLSTMConfig)
cs.store(group="model.optimizer", name="adam", node=AdamOptimizer)
cs.store(group="model.optimizer", name="adagrad", node=AdaGradOptimizer)

cs.store(group="logger", name="wandb", node=Wandb)
cs.store(group="pl_trainer", name="default", node=LightningTrainerConfig)

cs.store(name="config", node=TrainConfig)

