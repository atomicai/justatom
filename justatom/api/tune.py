import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Union

import dotenv
import numpy as np
import polars as pl
import pytorch_lightning as L
import torch
import torchmetrics as tm
from loguru import logger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import ConcatDataset
from transformers.optimization import Adafactor, AdafactorSchedule

import wandb
from justatom.configuring import Config

# Model IO and Prediction Head Flow
from justatom.modeling.head import ANNHead
from justatom.modeling.mask import ILanguageModel
from justatom.modeling.prime import E5Model
from justatom.processing import IProcessor, ITokenizer, igniset
from justatom.processing.loader import NamedDataLoader
from justatom.processing.prime import TRILMProcessor
from justatom.running.m1 import M1LMRunner
from justatom.tooling import stl
from justatom.tooling.stl import merge_in_order

# Training pipeline
from justatom.training.core import ILTrainer
from justatom.training.loss import TripletLoss

dotenv.load_dotenv()

logger.info(f"Enable MPS fallback = {os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')}")


def to_numpy(container):
    try:
        return container.cpu().numpy()
    except AttributeError:
        return container


def ignite_model_by(model_name_or_path, **props):
    if Path(model_name_or_path).exists():
        return ILanguageModel.load(model_name_or_path, **props)
    MAPPING = {"intfloat/multilingual-e5-base": E5Model}
    assert (
        model_name_or_path in MAPPING
    ), f"The model you want to initialize [{model_name_or_path}] is coming neither from huggingface hub nor locally saved"
    return MAPPING[model_name_or_path](**props)


def maybe_cuda_or_mps():
    if torch.cuda.is_available():
        return "cuda:0"
    elif torch.has_mps:
        return "mps"
    else:
        return "cpu"


def random_split(ds: ConcatDataset, lengths: List[int]):
    """
    Roughly split a Concatdataset into non-overlapping new datasets of given lengths.
    Samples inside Concatdataset should already be shuffled.

    :param ds: Dataset to be split.
    :param lengths: Lengths of splits to be produced.
    """
    if sum(lengths) != len(ds):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    try:
        import numpy as np

        idx_dataset = np.where(np.array(ds.cumulative_sizes) > lengths[0])[0][0]
    except IndexError:
        raise Exception(
            "All dataset chunks are being assigned to train set leaving no samples for dev set. "
            "Either consider increasing dev_split or setting it to 0.0\n"
            f"Cumulative chunk sizes: {ds.cumulative_sizes}\n"
            f"train/dev split: {lengths}"
        )

    assert idx_dataset >= 1, (
        "Dev_split ratio is too large, there is no data in train set. Please lower split =" f" {str(lengths)}"
    )

    train = ConcatDataset(ds.datasets[:idx_dataset])  # type: Dataset
    test = ConcatDataset(ds.datasets[idx_dataset:])  # type: Dataset
    return train, test


def check_and_raise(
    fpath: Union[str, Path],
    name: str = None,
    allowed_suffixes: Iterable[str] = (".csv"),
) -> bool:
    if fpath is None:
        return False
    suffixes = set(iter(allowed_suffixes))
    fp = Path(fpath)
    assert fp.exists(), f"Provided {name} dataset path {str(fp)} does not exsits"
    assert (
        fp.suffix in suffixes
    ), f"{name} dataset path extension {fp.suffix} is not yet supported. Please provide one of {' | '.join(allowed_suffixes)}"
    return fp


def ignite_loaders(
    processor: IProcessor,
    train_filepath: Union[str, Path],
    dev_filepath: Union[str, Path] = None,
    test_filepath: Union[str, Path] = None,
    split_ratio: float = None,
    shuffle: bool = True,
    content_key: str = "content",
    group_label_key: str = "group",
    batch_size: int = 4,
) -> Tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
]:
    train_filepath = Path(train_filepath) / "train.csv" if Path(train_filepath).is_dir() else train_filepath
    train_fpath = check_and_raise(train_filepath, "TRAIN", allowed_suffixes=[".csv"])

    logger.info(
        f"TRAINING [1/3] Using {str(train_fpath)} to perform training pipeline loading and converting to PyTorch dataset"
    )

    pl_train_view = pl.read_csv(train_fpath)

    train_docs = [
        {
            "content": x[content_key],
            "meta": {"group": hash(x[group_label_key]), "prefix": x.get("prefix", "")},
        }
        for x in pl_train_view.to_dicts()
    ]
    dataset, tensor_names = igniset(train_docs, processor=processor, batch_size=batch_size, shuffle=shuffle)

    split_ratio = 0.2 if dev_filepath is None and split_ratio is None else split_ratio

    dev_fpath = check_and_raise(dev_filepath, "DEV", allowed_suffixes=[".csv"])

    if dev_fpath:  # Implication is that `dev_fpath` exists hereby not equal to None => no need to split
        pl_dev_view = pl.read_csv(dev_fpath)
        dev_docs = [
            {
                "content": x[content_key],
                "meta": {
                    "group": hash(x[group_label_key]),
                    "prefix": x.get("prefix", ""),
                },
            }
            for x in pl_dev_view.to_dicts()
        ]
        dev_dataset, _ = igniset(dev_docs, processor=processor, batch_size=batch_size, shuffle=shuffle)
    else:  # Exception was not raised => perform split by ratio from train
        dev_size = int(len(dataset) * split_ratio)
        tra_size = len(dataset) - dev_size
        train_dataset, dev_dataset = random_split(dataset, [tra_size, dev_size])

    # NOTE: Use the same DEV dataset for TEST dataset without splitting if dataset_path is not defined.

    test_fpath = check_and_raise(test_filepath, "TEST", allowed_suffixes=[".csv"])

    if test_fpath:  # Exists and was not None
        pl_test_view = pl.read_csv(test_fpath)
        test_docs = [
            {
                "content": x[content_key],
                "meta": {
                    "group": hash(x[group_label_key]),
                    "prefix": x.get("prefix", ""),
                },
            }
            for x in pl_test_view.to_dicts()
        ]
        test_dataset, _ = igniset(test_docs, processor=processor, batch_size=batch_size, shuffle=shuffle)
    else:  # Was NONE
        test_dataset = dev_dataset

    # TODO: DRY. Replace boilerplate codes for <name>_fpath checking with one method.
    return (
        NamedDataLoader(train_dataset, batch_size=batch_size, tensor_names=tensor_names),
        NamedDataLoader(dev_dataset, batch_size=batch_size, tensor_names=tensor_names),
        NamedDataLoader(test_dataset, batch_size=batch_size, tensor_names=tensor_names),
        tensor_names,
    )


class ILRunner(L.LightningModule):
    def __init__(
        self,
        runner,
        loss,
        suffix: Iterable[str],
        label_suffix: str = "group_ids",
        grad_acc_steps: int = 6,
    ):
        super().__init__()
        self.runner = runner
        self.loss = loss
        self.grad_acc_steps = grad_acc_steps
        self.monitor = []
        self.suffix = set(suffix)
        self.label_suffix = label_suffix

    def configure_optimizers(self):
        optimizer = Adafactor(
            self.runner.parameters(),
            scale_parameter=True,
            relative_step=True,
            warmup_init=True,
            lr=None,
        )
        lr_scheduler = AdafactorSchedule(optimizer)

        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        # TODO: Fix Focal Loss ?
        xs = {k: batch[k].to(self.device) for k in batch if not k.endswith(self.label_suffix)}
        ys = {k: batch[k].to(self.device) for k in batch if k.endswith(self.label_suffix)}

        # TODO: How to perform `loss.FocalLoss` using this `group_ids` architecture ?

        # Maybe timeit below to log average RPS / FPS per different length sample

        output = self.runner(xs, average=True)  # num_heads x batch_size x embedding_dim

        # Maybe wrap around code below in `runner.logits_to_loss` ?

        all_losses = []

        for head, logits in zip(self.runner.prediction_heads, output):
            if isinstance(head.loss, TripletLoss):
                loss, info = head.loss(logits, ys.get("group_ids"))
                dist_p, dist_n, dist_acc = (
                    info.get("dist_p"),
                    info.get("dist_n"),
                    info.get("dist_acc"),
                )
                self.log("TrainingLoss", loss, logger=True)
                self.log("DistAcc", dist_acc, logger=True)
                self.log("POS distance", dist_p, logger=True)
                self.log("NEG distance", dist_n, logger=True)
            else:
                raise ValueError(f"Unexpected LOSS {self.loss} of UNKNOWN type for ANN tuning")
            all_losses.append(loss)
        per_sample_loss = self.runner.loss_aggregation_fn(all_losses)
        L = self.adjust_loss(per_sample_loss)

        return L

    def compute_metrics(self, head, logits, labels, metrics: Dict = None):
        # The `metrics` is supposed yet another mapping metrics.
        # Maps the returned keys from the head (e.g. loss) to the display name you want them to see.
        # For the `TripletLoss` you can add
        # `[dist_p, dist_n, nonzero_count]` one of those and map them to the nice format in logging.
        GRANTED_METRIC_NAME = dict(
            prec="Precision",
            dist_acc="DistanceAcc",
            dist_sm="DistanceSm",
            rel_dist="RelativeDistance",
        )
        metrics = merge_in_order(GRANTED_METRIC_NAME, metrics)
        if isinstance(head.loss, TripletLoss):
            _, info = head.loss(logits, labels)
        response = dict()
        for metric, display_metric in metrics.items():
            if metric in info:
                response[display_metric] = info[metric]
        return response

    @torch.no_grad
    def validation_step(self, batch, batch_idx):
        xs = {k: batch[k].to(self.device) for k in batch if not k.endswith(self.label_suffix)}
        ys = {k: batch[k].to(self.device) for k in batch if k.endswith(self.label_suffix)}
        output = self.runner(xs, average=True)  # num_heads x batch_size x embedding_dim

        for head, logits in zip(self.runner.prediction_heads, output):
            metrics = self.compute_metrics(head, logits, ys.get(self.label_suffix))
            for _metric, _score in metrics.items():
                self.log(_metric, _score)

    @torch.no_grad
    def test_step(self, batch, batch_idx):
        pass
        # Metrics ?
        # SEE
        # (a) prec
        # (b) dist_acc
        # (c) dist_sm
        # (d) nonzero_count
        # (e) rel_dist

    def adjust_loss(self, loss):
        mean_loss = loss.mean()
        if self.grad_acc_steps > 1:
            mean_loss = mean_loss / self.grad_acc_steps
        return mean_loss


def main(
    max_seq_len: int = None,
    index_name: str = None,
    dataset_path: str = None,
    shuffle: bool = None,
    model_name_or_path: str = None,
    model_props: Dict = None,
    loss: str = None,
    loss_props: Dict = None,
    batch_size: int = None,
    early_stopping_metric: str = None,
    early_stopping_mode: str = None,
    early_stopping_size: str = None,
    max_epochs: int = None,
    do_scale: bool = None,
    do_scale_unit: int = None,
    log_every_n_steps: int = None,
    save_top_k: int = None,
    devices: str = None,
    val_check_interval: int = None,
    save_model_path: str = None,
    opts: Dict = None,
):
    max_seq_len = max_seq_len or Config.train.max_seq_len
    index_name = index_name or Config.train.index_name
    dataset_path = dataset_path or Config.train.dataset_path
    _shuffle = Config.train.shuffle if shuffle is None else shuffle
    ### Load the model
    model_name_or_path = model_name_or_path or Config.train.model.model_name_or_path
    model_props = merge_in_order(model_props, Config.train.model.props)

    # model = ILanguageModel.load(model_name_or_path, max_seq_len=max_seq_len)
    model = ignite_model_by(model_name_or_path=model_name_or_path)
    # model = ILanguageModel.model(model_name_or_path, model_props)
    ### Training configuration
    loss = loss or Config.train.loss
    loss_props = merge_in_order(loss_props, Config.train.loss_props)
    batch_size = batch_size or Config.train.batch_size
    early_stopping_metric = early_stopping_metric or Config.train.early_stopping.metric
    early_stopping_mode = early_stopping_mode or Config.train.early_stopping.mode
    early_stopping_size = int(early_stopping_size or Config.train.early_stopping.size)
    max_epochs = max_epochs or Config.train.max_epochs
    # Add scaling
    do_scale, do_scale_unit = do_scale or Config.train.do_scale, int(do_scale_unit or Config.train.do_scale_unit)
    if do_scale is True and not do_scale_unit:
        do_scale_unit = 1.0
    else:
        do_scale_unit *= 1.0
    # TODO: maybe_scale()

    log_every_n_steps = int((log_every_n_steps or Config.train.log_every_n_steps) / do_scale_unit)
    save_top_k = save_top_k or Config.train.save_top_k
    devices = devices or Config.train.devices
    val_check_interval = int((val_check_interval or Config.train.val_check_interval) / do_scale_unit)
    save_model_path = Path(save_model_path or Config.train.save_model_path)
    # Snapshot for naming given correct hyperparams
    snap_opts = merge_in_order(loss_props, opts)
    snap_name = stl.snapshot(snap_opts)
    snap_model_path = Path(os.getcwd()) / save_model_path / snap_name
    # For now, we only switch between different `margin` aka `loss_props` =>
    # Hence for every unique `loss_props` we could generate unique path for `EarlyStopping`
    # TODO: Add other props via general `opts: Dict` => `merge_in_order(opts, loss_props)` =>
    # and make snapshot of `opts`
    tokenizer = ITokenizer.from_pretrained(model_name_or_path)
    processor = TRILMProcessor(tokenizer=tokenizer, max_seq_len=max_seq_len)
    device = maybe_cuda_or_mps()
    runner = M1LMRunner(
        model=model,
        prediction_heads=[ANNHead(loss_fn=loss, device=device, **loss_props)],
        device=device,
    )
    pl_runner = ILRunner(runner=runner, loss=None, suffix=["group_ids"])
    logger.info(
        f'All components (1) [Model {model.__class__.__name__}] (2) [Head(s) "{len(runner.prediction_heads)}" - {runner.prediction_heads[0].__class__.__name__}] (3) Loss {runner.prediction_heads[0].loss}'
    )

    train_loader, dev_loader, test_loader, tensor_names = ignite_loaders(
        processor=processor,
        train_filepath=Path(os.getcwd()) / dataset_path,
        batch_size=batch_size,
        shuffle=_shuffle,
    )

    # ML Logger
    ml_logger = WandbLogger(project="POLAROIDS.AI", name=f"LOSS={loss.upper()} {snap_name}")

    # Callback(s)...
    es_callback = EarlyStopping(monitor=early_stopping_metric, patience=early_stopping_size)
    mc_callback = ModelCheckpoint(
        monitor=early_stopping_metric,
        mode=early_stopping_mode,
        save_top_k=save_top_k,
        dirpath=str(snap_model_path),
        auto_insert_metric_name=True,
        every_n_train_steps=1,
    )

    pipeline = ILTrainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices="auto",
        callbacks=[es_callback, mc_callback],
        logger=ml_logger,
        val_check_interval=min(val_check_interval, len(train_loader)),
        log_every_n_steps=log_every_n_steps,
        # limit_train_batches=300,
        # limit_val_batches=100
    )

    # Early stopping on which metric ?

    pipeline.fit(model=pl_runner, train_dataloaders=train_loader, val_dataloaders=dev_loader)
    pipeline.test(model=pl_runner, dataloaders=test_loader)

    # Save the model and (index) ?
    logger.info(f"{mc_callback.best_model_path}")
    logger.info(f"{mc_callback.best_model_score}")
    wandb.finish()


if __name__ == "__main__":
    for bs in (128, 256):
        for margin in np.arange(0.4, 1.5, 0.4):
            main(
                loss="triplet",
                shuffle=True,
                do_scale=True,
                do_scale_unit=bs * 1.0 / 32,
                loss_props={"margin": np.round(margin, 2)},
                opts={"bs": bs, "esm": "DistanceAcc", "dataset": "polaroids.ai"},
                batch_size=bs,
            )
