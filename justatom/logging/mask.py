import abc

from loguru import logger

from justatom.etc.visual import WELCOME_MSG


class ILogger(abc.ABC):  # noqa: B024
    """
    Base class for tracking experiments.

    This class can be extended to implement custom logging backends like MLFlow, Tensorboard, or WANDB.
    """

    disable_logging = False

    def __init__(self, log_batch_metrics: bool, log_epoch_metrics: bool, tracking_uri: bool | None = False, **kwargs):
        self.tracking_uri = tracking_uri
        self.log_batch_metrics = log_batch_metrics
        self.log_epoch_metrics = log_epoch_metrics
        logger.success(WELCOME_MSG)

    def log_metrics(self, metrics, step, **kwargs):
        raise NotImplementedError()

    @classmethod
    def log_artifacts(cls, self):
        raise NotImplementedError()

    @classmethod
    def log_params(self, params, **kwars):
        raise NotImplementedError()
