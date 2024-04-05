from pathlib import Path
import os
from loguru import logger
import lightning as L
from justatom.etc import delete_folder


class ILTrainer(L.Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy.remove_checkpoint = self.remove_checkpoint

    def save_checkpoint(self, filepath, weights_only=False):
        if self.is_global_zero:
            dirpath = os.path.split(filepath)[0]
            runner = self.model
            # runner have model and heads.
            save_dir = Path(dirpath) / f"epoch={self.current_epoch}-step={self.global_step}"
            save_dir.mkdir(parents=True, exist_ok=True)
            runner.runner.save(save_dir)
            logger.info(f"Saving checkpoint epoch={self.current_epoch}-step={self.global_step}")

    def remove_checkpoint(self, filepath):
        if self.is_global_zero:
            # filepath = -> оканчивается на .ckpt
            fp = Path(filepath)
            data_dir = fp.parent / fp.stem
            delete_folder(data_dir)
            logger.info(f"Removing checkpoint epoch={self.current_epoch}-step={self.global_step}")


class ILRunner(L.LightningModule):
    pass


__all__ = ["ILTrainer", "ILRunner"]
