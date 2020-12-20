from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from torchtree.data.sst import SSTDatamodule
from torchtree.data.vocabulary import Vocabulary
from torchtree.utils import get_project_root
from treemodels.treelstm.pl_module import TreeLSTMClassification

from treemodels.config import TrainConfig
import hydra
from omegaconf import OmegaConf

from hydra.core.config_store import ConfigStore


@hydra.main(config_path="conf", config_name="config")
def train(cfg: TrainConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    pl.seed_everything(cfg.seed)

    # load data and vocabulary
    data_dir = Path(cfg.dataset.data_dir)


    # init loggers
    wandb_logger = WandbLogger(
        # offline=False,
        offline=cfg.wandb_offline,
        project='tree-lstm'
    )

    # init model
    if not data_dir.exists() or cfg.dataset.force_reload:
        data_module: pl.LightningDataModule = hydra.utils.instantiate(cfg.dataset, force_reload=cfg.dataset.force_reload)
        data_module.prepare_data()

    vocab_file = data_dir / "vocab.txt"
    vocab = Vocabulary.load(vocab_file)

    model = hydra.utils.instantiate(cfg.model, vocabulary=vocab)

    print(model)
    trainer = pl.Trainer(
        min_epochs=cfg.min_epochs,
        max_epochs=cfg.max_epochs,
        logger=wandb_logger,
        gpus=cfg.gpus,
    )

    trainer.fit(model, datamodule=data_module)
    trainer.test()


if __name__ == '__main__':
    train()