from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from torchtree.data.vocabulary import Vocabulary

from treemodels.config import TrainConfig
import hydra
from omegaconf import OmegaConf

import wandb


@hydra.main(config_path="conf", config_name="config")
def train(cfg: TrainConfig) -> None:
    print("Running with the following config:")
    print(OmegaConf.to_yaml(cfg))

    # for hparam sweep
    wandb.init(config=cfg)
    cfg = wandb.config

    pl.seed_everything(cfg.seed)

    # load data and vocabulary
    data_dir = Path(cfg.dataset.data_dir)


    # init data
    data_module: pl.LightningDataModule = hydra.utils.instantiate(
        cfg.dataset, force_reload=cfg.dataset.force_reload,
        train_batch_size=cfg.train_batch_size,
        eval_batch_size=cfg.eval_batch_size,
    )
    if not data_dir.exists() or cfg.dataset.force_reload:
        data_module.prepare_data()

    # load vocab separately
    vocab_file = data_dir / "vocab.txt"
    vocab = Vocabulary.load(vocab_file)


    model = hydra.utils.instantiate(
        cfg.model,
        num_classes=data_module.class_label.num_classes,
        vocabulary=vocab,
        cfg=cfg
    )

    # init loggers
    logger = hydra.utils.instantiate(cfg.logger)

    # somehow this not works with hydra instantiate
    trainer = pl.Trainer(
        logger=logger,

        # default_root_dir=cfg.pl_trainer.default_root_dir,
        gradient_clip_val=cfg.pl_trainer.gradient_clip_val,
        gpus=cfg.pl_trainer.gpus,
        auto_select_gpus=cfg.pl_trainer.auto_select_gpus,
        log_gpu_memory=cfg.pl_trainer.log_gpu_memory,
        overfit_batches=cfg.pl_trainer.overfit_batches,
        track_grad_norm=cfg.pl_trainer.track_grad_norm,
        accumulate_grad_batches=cfg.pl_trainer.accumulate_grad_batches,
        max_epochs=cfg.pl_trainer.max_epochs,
        min_epochs=cfg.pl_trainer.min_epochs,
        max_steps=cfg.pl_trainer.max_steps,
        min_steps=cfg.pl_trainer.min_steps,
        limit_train_batches=cfg.pl_trainer.limit_train_batches,
        limit_val_batches=cfg.pl_trainer.limit_val_batches,
        limit_test_batches=cfg.pl_trainer.limit_test_batches,
        val_check_interval=cfg.pl_trainer.val_check_interval,
        precision=cfg.pl_trainer.precision,
        resume_from_checkpoint=cfg.pl_trainer.resume_from_checkpoint,
        deterministic=cfg.pl_trainer.deterministic,
        reload_dataloaders_every_epoch=cfg.pl_trainer.reload_dataloaders_every_epoch,
        auto_lr_find=cfg.pl_trainer.auto_lr_find,
        auto_scale_batch_size=cfg.pl_trainer.auto_scale_batch_size,
    )

    trainer.fit(model, datamodule=data_module)
    # trainer.test()


if __name__ == '__main__':
    wandb.agent(sweep_id, function=train)
    train()