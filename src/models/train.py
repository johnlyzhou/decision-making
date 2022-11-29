from pathlib import Path

from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint, EarlyStopping


def train(system_class: LightningModule,
          config: dict,
          experiment_dir: str = "experiments",
          checkpoint_name: str = "vae_{val_loss:.3f}.ckpt"):

    seed_everything(config["random_seed"])
    system = system_class(config)

    experiment_dir = Path(f"{experiment_dir}")
    experiment_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{experiment_dir}/models",
        filename=checkpoint_name,
        auto_insert_metric_name=True
    )

    trainer = Trainer(
        **config["trainer"],
        callbacks=[
            checkpoint_callback,
            EarlyStopping(monitor="val_loss", patience=3),
            TQDMProgressBar(refresh_rate=20)
        ],
        default_root_dir=str(experiment_dir)
    )
    trainer.fit(system)
    return system, trainer
