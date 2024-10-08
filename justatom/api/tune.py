import os
from collections.abc import Iterable
from pathlib import Path

import dotenv
import numpy as np
import polars as pl
import pytorch_lightning as L
import torch
import wandb
from loguru import logger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import ConcatDataset
from transformers.optimization import Adafactor, AdafactorSchedule

from justatom.configuring import Config
from justatom.etc.schema import Document

# Model IO and Prediction Head Flow
from justatom.modeling.head import ANNHead
from justatom.modeling.mask import ILanguageModel
from justatom.processing import IProcessor, ITokenizer, igniset
from justatom.processing.loader import NamedDataLoader
from justatom.processing.prime import ContrastiveProcessor, TripletProcessor
from justatom.running.m1 import M1LMRunner
from justatom.tooling import stl
from justatom.tooling.stl import merge_in_order

# Training pipeline
from justatom.training.core import ILTrainer
from justatom.training.loss import ContrastiveLoss, TripletLoss

dotenv.load_dotenv()

logger.info(f"Enable MPS fallback = {os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')}")



class ILDataModule(L.LightningDataModule):
    """
    Wrapper class for handling data loading with dynamic batch size.
    """
    def __init__(
        self,
        processor: IProcessor | None = None,
        train_filepath: str | Path = None,
        dev_filepath: str | Path = None,
        test_filepath: str | Path = None,
        split_ratio: float = None,
        shuffle: bool = True,
        group_field: str = None,  # group
        batch_size: int = 4,
        search_field: str = None,
        content_field: str = None,  # content
        prefix_field: str = None,
        prefix_search_field: str = None,
        prefix_content_field: str = None,
        filters: dict | None = None,
        dtypes: dict | None = None,
        dbs_epochs: list[int] | None = None,
        dbs_batch_sizes: list[int]| None = None,
    ):
        super().__init__()
        dbs_epochs = dbs_epochs or []
        dbs_batch_sizes = dbs_batch_sizes or []
        self._dbs = {epoch: batch_size for epoch, batch_size in zip(dbs_epochs, dbs_batch_sizes, strict=False)}
        self.processor = processor
        self.train_filepath = train_filepath
        self.batch_size = self._dbs.get(min(dbs_epochs), 1) if self._dbs else batch_size
        self.search_field = search_field
        self.content_field = content_field
        self.group_field = group_field
        self.prefix_field = prefix_field
        self.prefix_search_field = prefix_search_field
        self.prefix_content_field = prefix_content_field
        self.shuffle = shuffle
        self.dtypes = dtypes
        self._train_dataloader, self._val_dataloader, self._test_dataloader, _ = self.ignite_loaders(self.batch_size)
        self.min_train_dataloader_len = self._min_train_dataloader_len()
        return

    def ignite_loaders(self, batch_size: int = 1):
        return ignite_loaders(
            processor=self.processor,
            train_filepath=self.train_filepath,
            batch_size=batch_size,
            search_field=self.search_field,
            content_field=self.content_field,
            group_field=self.group_field,
            prefix_field=self.prefix_field,
            prefix_search_field=self.prefix_search_field,
            prefix_content_field=self.prefix_content_field,
            shuffle=self.shuffle,
            dtypes=self.dtypes,
        )

    def train_dataloader(self):
        if self.trainer.current_epoch in self._dbs:
            self.batch_size = self._dbs[self.trainer.current_epoch]
            self._train_dataloader, _, _, _ = self.ignite_loaders(self.batch_size)
        return self._train_dataloader

    def val_dataloader(self):
        return self._val_dataloader

    def test_dataloader(self):
        return self._test_dataloader

    def _min_train_dataloader_len(self) -> int:
        return (
            len(self._train_dataloader) if not self._dbs
            else len(self._train_dataloader) * self.batch_size // max(self._dbs.values())
        )


def to_numpy(container):
    try:
        return container.cpu().numpy()
    except AttributeError:
        return container


def maybe_cuda_or_mps():
    if torch.cuda.is_available():
        return "cuda:0"
    elif torch.has_mps:
        return "mps"
    else:
        return "cpu"


def ignite_processor(loss_name: str, **props):
    MAPPING = {"contrastive": ContrastiveProcessor, "triplet": TripletProcessor}
    if loss_name.lower() not in MAPPING:
        msg = f"Specified loss {loss_name.upper()} is not available. Use one of the following {','.join(MAPPING.keys())}"
        logger.error(msg)
        raise ValueError(msg)
    return MAPPING.get(loss_name)(**props)


def random_split(ds: ConcatDataset, lengths: list[int]):
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
        raise Exception(  # noqa: B904
            "All dataset chunks are being assigned to train set leaving no samples for dev set. "
            "Either consider increasing dev_split or setting it to 0.0\n"
            f"Cumulative chunk sizes: {ds.cumulative_sizes}\n"
            f"train/dev split: {lengths}"
        )

    assert idx_dataset >= 1, "Dev_split ratio is too large, there is no data in train set. Please lower split =" f" {str(lengths)}"

    train = ConcatDataset(ds.datasets[:idx_dataset])  # type: Dataset
    test = ConcatDataset(ds.datasets[idx_dataset:])  # type: Dataset
    return train, test


def check_and_raise(
    fpath: str | Path,
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
    return True


def check_structure_and_raise_with_prepare(
    pl_data: pl.DataFrame,
    processor: IProcessor,
    search_field: str | None = None,
    content_field: str | None = None,
    group_field: str | None = None,
    prefix_field: str | None = None,
    prefix_search_field: str | None = None,
    prefix_content_field: str | None = None,
) -> pl.DataFrame:
    if search_field is not None:
        assert search_field in pl_data.columns, f"Search field [{search_field}] is not present within dataset."
    if content_field is not None:
        assert content_field in pl_data.columns, f"Content field [{content_field}] is not present within dataset."
    if group_field is not None:
        assert group_field in pl_data.columns, f"Group field [{group_field}] is not present within dataset."
    if prefix_field is not None:
        assert prefix_field in pl_data.columns, f"Prefix field [{prefix_field}] is not present within dataset."
    js_data: list[dict] = None
    # Make sure that the `processor` has correct type
    if search_field is not None and content_field is not None:
        assert isinstance(
            processor, ContrastiveProcessor
        ), f"You provided both `search_field`={search_field} and `content_field`={content_field} but processor is of wrong type = [{type(processor)}]"  # noqa: E501
        # TODO: Return
        if prefix_field is None and prefix_search_field is None and prefix_content_field is None:
            pl_data = pl_data.select([search_field, content_field])
            js_data = [dict(query=x.get(search_field), content=x.get(content_field)) for x in pl_data.to_dicts()]
        elif prefix_field is not None:
            pl_data = pl_data.select([search_field, content_field, prefix_field])
            js_data = [
                dict(
                    query=x.get(search_field),
                    content=x.get(content_field),
                    meta=dict(
                        queries_prefix=x.get(prefix_field),
                        pos_queries_prefix=x.get(prefix_field),
                    ),
                )
                for x in pl_data.to_dicts()
            ]
        else:
            assert (
                prefix_search_field is not None and prefix_content_field is not None
            ), "You seem to provide one of `prefix_search_field` or `prefix_content_field` but not both and at the same time, yet `prefix_field` is None"  # noqa: E501
            pl_data = pl_data.select([search_field, content_field, prefix_search_field, prefix_content_field])
            js_data = [
                dict(
                    query=x.get(search_field),
                    content=x.get(content_field),
                    meta=dict(
                        queries_prefix=x.get(prefix_search_field),
                        pos_queries_prefix=x.get(prefix_content_field),
                    ),
                )
                for x in pl_data.to_dicts()
            ]
    elif content_field is not None and group_field is not None:
        assert isinstance(
            processor, TripletProcessor
        ), f"You provided both `content_field`={content_field} and `group_field`={group_field} but processor is of wrong type = [{type(processor)}]"  # noqa: E501
        pl_data = (
            pl_data.select([content_field, group_field])
            if prefix_field is None
            else pl_data.select([content_field, group_field, prefix_field])
        )
        js_data = [
            Document.from_dict(
                dict(
                    content=x.get(content_field),
                    prefix=x.get(prefix_field, ""),
                    group=hash(x.get(group_field)),
                )
            ).to_dict()
            for x in pl_data.to_dicts()
        ]
        # TODO: Return
    else:
        msg = "You should either set `search_field` and `content_field` or `content_field` and `group_label_field`"
        logger.error(msg)
        raise ValueError(msg)
    return pl_data, js_data


def check_and_filter(pl_data: pl.DataFrame, filters: dict | None = None):
    if filters is not None:
        fields = filters.get("fields", [])
        for field in fields:
            pl_data = pl_data.filter(pl.col(field).is_not_null())
    return pl_data


def check_and_scan(
    filepath,
    allowed_suffixes,
    split_name: str,
    dtypes: dict | None = None,
    strict_raise: bool = True,
):
    if not check_and_raise(filepath, name=split_name, allowed_suffixes=allowed_suffixes):
        if strict_raise:
            msg = "Filepath you have set is None. `strict_raise`=True by default. Either turn it off or provide valid path."
            logger.error(msg)
            raise ValueError(msg)
        else:
            return None
    fpath = Path(filepath)
    if fpath.suffix in [".csv", ".xlsx"]:
        pl_view = pl.read_csv(fpath, dtypes=dtypes) if fpath.suffix == ".csv" else pl.read_excel(fpath)
    elif fpath.suffix in [".json", ".jsonl"]:
        pl_view = pl.read_json(fpath)
    else:
        if strict_raise:
            msg = f"The following filepath {fpath} has unsupported [{fpath.suffix}] suffix"
            logger.error(msg)
            raise ValueError(msg)
        else:
            return None
    return pl_view


def ignite_loaders(
    processor: IProcessor,
    train_filepath: str | Path,
    dev_filepath: str | Path = None,
    test_filepath: str | Path = None,
    split_ratio: float = None,
    shuffle: bool = True,
    group_field: str = None,  # group
    batch_size: int = 4,
    search_field: str = None,
    content_field: str = None,  # content
    prefix_field: str = None,
    prefix_search_field: str = None,
    prefix_content_field: str = None,
    filters: dict | None = None,
    dtypes: dict | None = None,
) -> tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
]:
    train_filepath = Path(train_filepath) / "train.csv" if Path(train_filepath).is_dir() else train_filepath

    pl_train_view = check_and_scan(
        train_filepath,
        allowed_suffixes=[".csv", ".xlsx", ".json", ".jsonl"],
        split_name="TRAIN",
        dtypes=dtypes,
    )

    logger.info(
        f"TRAINING [1/3] Using {str(train_filepath)} to perform training pipeline loading and converting to PyTorch dataset"
    )

    pl_train_view = check_and_filter(pl_train_view, filters=filters)

    pl_train_view, js_train_docs = check_structure_and_raise_with_prepare(
        pl_train_view,
        processor=processor,
        search_field=search_field,
        content_field=content_field,
        group_field=group_field,
        prefix_field=prefix_field,
        prefix_search_field=prefix_search_field,
        prefix_content_field=prefix_content_field,
    )

    dataset, tensor_names = igniset(js_train_docs, processor=processor, batch_size=batch_size, shuffle=shuffle)

    split_ratio = 0.2 if dev_filepath is None and split_ratio is None else split_ratio

    pl_dev_view = check_and_scan(
        dev_filepath,
        allowed_suffixes=[".csv", ".xlsx", ".json", ".jsonl"],
        split_name="DEV",
        dtypes=dtypes,
        strict_raise=False,
    )

    if pl_dev_view is not None:
        logger.info(
            f"TRAINING [2/3] Using {str(dev_filepath)} to perform training pipeline loading and converting to PyTorch DEV dataset"
        )
        pl_dev_view = check_and_filter(pl_dev_view, dtypes=dtypes)
        pl_dev_view, js_dev_docs = check_structure_and_raise_with_prepare(
            pl_dev_view,
            processor=processor,
            search_field=search_field,
            content_field=content_field,
            group_field=group_field,
            prefix_field=prefix_field,
            prefix_search_field=prefix_search_field,
            prefix_content_field=prefix_content_field,
        )
        dev_dataset, _ = igniset(js_dev_docs, processor=processor, batch_size=batch_size, shuffle=shuffle)
    else:  # Exception was not raised => perform split by ratio from train
        logger.info(
            f"TRAINING [2/3] Using SPLIT ration of {str(split_ratio)} to perform training pipeline loading and converting to PyTorch DEV dataset"  # noqa: E501
        )
        dev_size = int(len(dataset) * split_ratio)
        tra_size = len(dataset) - dev_size
        train_dataset, dev_dataset = random_split(dataset, [tra_size, dev_size])

    # NOTE: Use the same DEV dataset for TEST dataset without splitting if dataset_path is not defined.

    pl_test_view = check_and_scan(
        test_filepath,
        allowed_suffixes=[".csv", ".xlsx", ".json", ".jsonl"],
        split_name="TEST",
        dtypes=dtypes,
        strict_raise=False,
    )

    if pl_test_view is not None:  # Exists and was not None
        logger.info(
            f"TRAINING [3/3] Using {str(test_filepath)} to perform training pipeline loading and converting to PyTorch TEST dataset"
        )
        pl_test_view = check_and_filter(pl_dev_view, dtypes=dtypes)
        pl_test_view, js_test_docs = check_structure_and_raise_with_prepare(
            pl_test_view,
            search_field=search_field,
            content_field=content_field,
            group_field=group_field,
            prefix_field=prefix_field,
            prefix_search_field=prefix_search_field,
            prefix_content_field=prefix_content_field,
        )
        test_dataset, _ = igniset(js_test_docs, processor=processor, batch_size=batch_size, shuffle=shuffle)
    else:  # Was NONE
        test_dataset = dev_dataset
        logger.info("TRAINING [3/3] Using DEV split to perform training pipeline loading and converting to PyTorch TEST dataset")

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

        for head in self.runner.prediction_heads:
            if isinstance(head.loss, TripletLoss):
                loss, info = head.loss(*output, ys.get("group_ids"))
                dist_p, dist_n, dist_acc = (
                    info.get("dist_p"),
                    info.get("dist_n"),
                    info.get("dist_acc"),
                )
                self.log("TrainingLoss", loss, logger=True)
                self.log("DistAcc", dist_acc, logger=True)
                self.log("POS distance", dist_p, logger=True)
                self.log("NEG distance", dist_n, logger=True)
            elif isinstance(head.loss, ContrastiveLoss):
                loss = head.loss(*output)
                # (1) output[0] -> queries.
                # (2) output[1] -> pos_queries.
                # (3) output[2] -> neg_queries
                self.log("TrainingLoss", loss, logger=True)
            else:
                raise ValueError(f"Unexpected LOSS {self.loss} of UNKNOWN type for ANN tuning")
            all_losses.append(loss)
        per_sample_loss = self.runner.loss_aggregation_fn(all_losses)
        L = self.adjust_loss(per_sample_loss)

        return L

    def compute_metrics(self, head, logits, labels, metrics: dict = None):
        # The `metrics` is supposed yet another mapping metrics.
        # Maps the returned keys from the head (e.g. loss) to the display name you want them to see.
        # For the `TripletLoss` you can add
        # `[dist_p, dist_n, nonzero_count]` one of those and map them to the nice format in logging.
        GRANTED_TRIPLET_METRIC_NAME = dict(
            prec="Precision",
            dist_acc="DistanceAcc",
            dist_sm="DistanceSm",
            rel_dist="RelativeDistance",
        )
        GRANTED_CONTRASTIVE_METRIC_NAME = dict(top1="HitRate")  # noqa: F841
        metrics = dict()
        response = dict()
        if isinstance(head.loss, TripletLoss):
            _, info = head.loss(logits, labels)
            metrics = merge_in_order(GRANTED_TRIPLET_METRIC_NAME, metrics)
        elif isinstance(head.loss, ContrastiveLoss):
            # metrics = merge_in_order(GRANTED_CONTRASTIVE_METRIC_NAME, metrics)
            metrics = dict()
        for metric, display_metric in metrics.items():
            if metric in info:
                response[display_metric] = info[metric]
        return response

    @torch.no_grad
    def validation_step(self, batch, batch_idx):
        xs = {k: batch[k].to(self.device) for k in batch if not k.endswith(self.label_suffix)}
        ys = {k: batch[k].to(self.device) for k in batch if k.endswith(self.label_suffix)}
        output = self.runner(xs, average=True)  # num_heads x batch_size x embedding_dim

        for head, logits in zip(self.runner.prediction_heads, output, strict=False):
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
    search_field: str = None,
    content_field: str = None,
    group_field: str = None,
    prefix_field: str = None,
    prefix_search_field: str = None,
    prefix_content_field: str = None,
    shuffle: bool = None,
    model_name_or_path: str = None,
    model_props: dict = None,
    loss: str = None,
    loss_props: dict = None,
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
    opts: dict = None,
    dtypes: dict | None = None,
    dynamic_bs_epochs: list[int] | None = None,
    dynamic_bs_batch_sizes: list[int] | None = None,
):
    max_seq_len = max_seq_len or Config.train.max_seq_len
    index_name = index_name or Config.train.index_name
    dataset_path = dataset_path or Config.train.dataset_path
    _shuffle = Config.train.shuffle if shuffle is None else shuffle
    ### Load the model
    model_name_or_path = model_name_or_path or Config.train.model.model_name_or_path
    model_props = merge_in_order(model_props, Config.train.model.props)

    model = ILanguageModel.load(model_name_or_path, model_props)
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
    processor = ignite_processor(loss, tokenizer=tokenizer, max_seq_len=max_seq_len)
    device = maybe_cuda_or_mps()
    runner = M1LMRunner(
        model=model,
        processor=processor,
        prediction_heads=[ANNHead(loss_fn=loss, device=device, **loss_props)],
        device=device,
    )
    pl_runner = ILRunner(runner=runner, loss=None, suffix=["group_ids"])
    logger.info(
        f'All components (1) [Model {model.__class__.__name__}] (2) [Head(s) "{len(runner.prediction_heads)}" - {runner.prediction_heads[0].__class__.__name__}] (3) Loss {runner.prediction_heads[0].loss}'  # noqa: E501
    )

    datamodule = ILDataModule(
        processor=processor,
        train_filepath=Path(os.getcwd()) / dataset_path,
        batch_size=batch_size,
        search_field=search_field,
        content_field=content_field,
        group_field=group_field,
        prefix_field=prefix_field,
        prefix_search_field=prefix_search_field,
        prefix_content_field=prefix_content_field,
        shuffle=_shuffle,
        dtypes=dtypes,
        dbs_epochs=dynamic_bs_epochs,
        dbs_batch_sizes=dynamic_bs_batch_sizes,
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
        val_check_interval=min(val_check_interval, datamodule.min_train_dataloader_len),
        log_every_n_steps=log_every_n_steps,
        reload_dataloaders_every_n_epochs=1,
        # limit_train_batches=300,
        # limit_val_batches=100
    )

    # Early stopping on which metric ?

    pipeline.fit(model=pl_runner, datamodule=datamodule)
    pipeline.test(model=pl_runner, dataloaders=datamodule.test_dataloader())

    # Save the model and (index) ?
    logger.info(f"{mc_callback.best_model_path}")
    logger.info(f"{mc_callback.best_model_score}")
    wandb.finish()


if __name__ == "__main__":
    for bs in [16]:
        for margin in np.arange(0.4, 1.5, 0.4):
            main(
                dataset_path=str(Path(os.getcwd()) / ".data"),
                loss="contrastive",
                search_field="query",
                content_field="content",
                prefix_search_field="query_prefix",
                prefix_content_field="content_prefix",
                group_field=None,
                prefix_field=None,
                shuffle=True,
                do_scale=True,
                do_scale_unit=bs * 1.0 / 32,
                loss_props={"margin": np.round(margin, 2)},
                opts={
                    "bs": bs,
                    "model": "E5Base",
                    "esm": "TrainingLoss",
                    "dataset": "polaroids.ai",
                },
                batch_size=bs,
                dtypes={"group": str},
            )
