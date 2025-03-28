import inspect
import logging
import sys
from importlib import import_module

import torch
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel

# Used indirectly in _get_optim() to avoid name collision with torch's AdamW

try:
    from apex import amp

    try:
        from apex.parallel import convert_syncbn_model

        APEX_PARALLEL_AVAILABLE = True
    except AttributeError:
        APEX_PARALLEL_AVAILABLE = False
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False
    APEX_PARALLEL_AVAILABLE = False

from farm.utils import WANDBLogger as MlLogger

logger = logging.getLogger(__name__)


class WrappedDataParallel(DataParallel):
    """
    A way of adapting attributes of underlying class to parallel mode. See: https://pytorch.org/tutorials/beginner/former_torchies/parallelism_tutorial.html#dataparallel

    Gets into recursion errors. Workaround see: https://discuss.pytorch.org/t/access-att-of-model-wrapped-within-torch-nn-dataparallel-maximum-recursion-depth-exceeded/46975
    """

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class WrappedDDP(DistributedDataParallel):
    """
    A way of adapting attributes of underlying class to distributed mode. Same as in WrappedDataParallel above.
    Even when using distributed on a single machine with multiple GPUs, apex can speed up training significantly.
    Distributed code must be launched with "python -m torch.distributed.launch --nproc_per_node=1 run_script.py"
    """

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def initialize_optimizer(
    model,
    n_batches,
    n_epochs,
    device,
    learning_rate,
    optimizer_opts=None,
    schedule_opts=None,
    distributed=False,
    grad_acc_steps=1,
    local_rank=-1,
    use_amp=None,
):
    """
    Initializes an optimizer, a learning rate scheduler and converts the model if needed (e.g for mixed precision).
    Per default, we use transformers' AdamW and a linear warmup schedule with warmup ratio 0.1.
    You can easily switch optimizer and schedule via `optimizer_opts` and `schedule_opts`.

    :param model: model to optimize (e.g. trimming weights to fp16 / mixed precision)
    :type model: AdaptiveModel
    :param n_batches: number of batches for training
    :type n_batches: int
    :param n_epochs: number of epochs for training
    :param device:
    :param learning_rate: Learning rate
    :type learning_rate: float
    :param optimizer_opts: Dict to customize the optimizer. Choose any optimizer available from torch.optim, apex.optimizers or
                           transformers.optimization by supplying the class name and the parameters for the constructor.
                           Examples:
                           1) AdamW from Transformers (Default):
                           {"name": "TransformersAdamW", "correct_bias": False, "weight_decay": 0.01}
                           2) SGD from pytorch:
                           {"name": "SGD", "momentum": 0.0}
                           3) FusedLAMB from apex:
                           {"name": "FusedLAMB", "bias_correction": True}
    :param schedule_opts: Dict to customize the learning rate schedule.
                          Choose any Schedule from Pytorch or Huggingface's Transformers by supplying the class name
                          and the parameters needed by the constructor.
                          If the dict does not contain ``num_training_steps`` it will be set by
                          calculating it from ``n_batches``, ``grad_acc_steps`` and ``n_epochs``.
                          Examples:
                          1) Linear Warmup (Default):
                          {"name": "LinearWarmup",
                          "num_warmup_steps": 0.1 * num_training_steps,
                          "num_training_steps": num_training_steps}
                          2) CosineWarmup:
                          {"name": "CosineWarmup",
                          "num_warmup_steps": 0.1 * num_training_steps,
                          "num_training_steps": num_training_steps}
                          3) CyclicLR from pytorch:
                          {"name": "CyclicLR", "base_lr": 1e-5, "max_lr":1e-4, "step_size_up": 100}
    :param distributed: Whether training on distributed machines
    :param grad_acc_steps: Number of steps to accumulate gradients for. Helpful to mimic large batch_sizes on small machines.
    :param local_rank: rank of the machine in a distributed setting
    :param use_amp: Optimization level of nvidia's automatic mixed precision (AMP). The higher the level, the faster the model.
                    Options:
                    "O0" (Normal FP32 training)
                    "O1" (Mixed Precision => Recommended)
                    "O2" (Almost FP16)
                    "O3" (Pure FP16).
                    See details on: https://nvidia.github.io/apex/amp.html
    :return: model, optimizer, scheduler
    """

    if use_amp and not AMP_AVAILABLE:
        raise ImportError(
            f"Got use_amp = {use_amp}, but cannot find apex. "
            "Please install Apex if you want to make use of automatic mixed precision. "
            "https://github.com/NVIDIA/apex"
        )

    if (schedule_opts is not None) and (not isinstance(schedule_opts, dict)):
        raise TypeError("Parameter schedule_opts must be None or " f"an instance of dict but was {type(schedule_opts)}!")

    num_train_optimization_steps = int(n_batches / grad_acc_steps) * n_epochs

    # Use some defaults to simplify life of inexperienced users
    if optimizer_opts is None:
        optimizer_opts = {"name": "TransformersAdamW", "correct_bias": False, "weight_decay": 0.01}
    optimizer_opts["lr"] = learning_rate

    if schedule_opts is None:
        # Default schedule: Linear Warmup with 10% warmup
        schedule_opts = {
            "name": "LinearWarmup",
            "num_warmup_steps": 0.1 * num_train_optimization_steps,
            "num_training_steps": num_train_optimization_steps,
        }

        # schedule_opts = {"name": "OneCycleLR", "max_lr":learning_rate, "pct_start": 0.1,
        #                  "total_steps": num_train_optimization_steps }
    elif "num_training_steps" not in schedule_opts:
        schedule_opts["num_training_steps"] = num_train_optimization_steps

    # Log params
    MlLogger.log_params(
        {
            "use_amp": use_amp,
            "num_train_optimization_steps": schedule_opts["num_training_steps"],
        }
    )

    # Get optimizer from pytorch, transformers or apex
    optimizer = _get_optim(model, optimizer_opts)

    # Adjust for parallel training + amp
    model, optimizer = optimize_model(model, device, local_rank, optimizer, distributed, use_amp)

    # Get learning rate schedule - moved below to supress warning
    scheduler = get_scheduler(optimizer, schedule_opts)

    return model, optimizer, scheduler


def _get_optim(model, opts):
    """Get the optimizer based on dictionary with options. Options are passed to the optimizer constructor.

    :param model: model to optimize
    :param opts: config dictionary that will be passed to optimizer together with the params
    (e.g. lr, weight_decay, correct_bias ...). no_decay' can be given - parameters containing any of those strings
    will have weight_decay set to 0.
    :return: created optimizer
    """

    optimizer_name = opts.pop("name", None)

    # Logging
    logger.info(f"Loading optimizer `{optimizer_name}`: '{opts}'")
    MlLogger.log_params(opts)
    MlLogger.log_params({"optimizer_name": optimizer_name})

    weight_decay = opts.pop("weight_decay", None)
    no_decay = opts.pop("no_decay", None)

    if no_decay:
        optimizable_parameters = [
            {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad], **opts},
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": 0.0,
                **opts,
            },
        ]
    else:
        optimizable_parameters = [{"params": [p for p in model.parameters() if p.requires_grad], **opts}]

    # default weight decay is not the same for all optimizers, so we can't use default value
    # only explicitly add weight decay if it's given
    if weight_decay is not None:
        optimizable_parameters[0]["weight_decay"] = weight_decay

    # Import optimizer by checking in order: torch, transformers, apex and local imports
    try:
        optim_constructor = getattr(import_module("torch.optim"), optimizer_name)
    except AttributeError:
        try:
            optim_constructor = getattr(import_module("transformers.optimization"), optimizer_name)
        except AttributeError:
            try:
                optim_constructor = getattr(import_module("apex.optimizers"), optimizer_name)
            except (AttributeError, ImportError):
                try:
                    # Workaround to allow loading AdamW from transformers
                    # pytorch > 1.2 has now also a AdamW (but without the option to set bias_correction = False,
                    # which is done in the original BERT implementation)
                    optim_constructor = getattr(sys.modules[__name__], optimizer_name)
                except (AttributeError, ImportError):
                    raise AttributeError(  # noqa: B904
                        f"Optimizer '{optimizer_name}' not found in 'torch', 'transformers', 'apex' or 'local imports"
                    )

    return optim_constructor(optimizable_parameters)


def get_scheduler(optimizer, opts):
    """Get the scheduler based on dictionary with options. Options are passed to the scheduler constructor.

    :param optimizer: optimizer whose learning rate to control
    :param opts: dictionary of args to be passed to constructor of schedule
    :return: created scheduler
    """
    schedule_name = opts.get("name")
    try:
        sched_constructor = getattr(import_module("torch.optim.lr_scheduler"), schedule_name)
    except AttributeError:
        try:
            # The method names in transformers became quite long and unhandy.
            # for convenience we offer usage of shorter alias (e.g. "LinearWarmup")
            scheduler_translations = {
                "LinearWarmup": "get_linear_schedule_with_warmup",
                "ConstantWarmup": "get_constant_schedule_with_warmup",
                "Constant": "get_constant_schedule",
                "CosineWarmup": "get_cosine_schedule_with_warmup",
                "CosineWarmupWithRestarts": "get_cosine_with_hard_restarts_schedule_with_warmup",
            }
            if schedule_name in scheduler_translations:
                schedule_name = scheduler_translations[schedule_name]
            # in contrast to torch, we actually get here a method and not a class
            sched_constructor = getattr(import_module("transformers.optimization"), schedule_name)
        except AttributeError:
            raise AttributeError(f"Scheduler '{schedule_name}' not found in 'torch' or 'transformers'")  # noqa: B904

    logger.info(f"Using scheduler '{schedule_name}'")

    # get supported args of constructor
    allowed_args = inspect.signature(sched_constructor).parameters.keys()

    # convert from warmup proportion to steps if required
    if "num_warmup_steps" in allowed_args and "num_warmup_steps" not in opts and "warmup_proportion" in opts:
        opts["num_warmup_steps"] = int(opts["warmup_proportion"] * opts["num_training_steps"])
        MlLogger.log_params({"warmup_proportion": opts["warmup_proportion"]})

    # only pass args that are supported by the constructor
    constructor_opts = {k: v for k, v in opts.items() if k in allowed_args}

    # Logging
    logger.info(f"Loading schedule `{schedule_name}`: '{constructor_opts}'")
    MlLogger.log_params(constructor_opts)
    MlLogger.log_params({"schedule_name": schedule_name})

    scheduler = sched_constructor(optimizer, **constructor_opts)
    scheduler.opts = opts  # save the opts with the scheduler to use in load/save
    return scheduler


def optimize_model(model, device, local_rank, optimizer=None, distributed=False, use_amp=None):
    """
    Wraps MultiGPU or distributed usage around a model
    No support for ONNX models

    :param model: model to optimize (e.g. trimming weights to fp16 / mixed precision)
    :type model: AdaptiveModel
    :param device: either gpu or cpu, get the device from initialize_device_settings()
    :param distributed: Whether training on distributed machines
    :param local_rank: rank of the machine in a distributed setting
    :param use_amp: Optimization level of nvidia's automatic mixed precision (AMP). The higher the level, the faster the model.
                    Options:
                    "O0" (Normal FP32 training)
                    "O1" (Mixed Precision => Recommended)
                    "O2" (Almost FP16)
                    "O3" (Pure FP16).
                    See details on: https://nvidia.github.io/apex/amp.html
    :return: model, optimizer
    """
    model, optimizer = _init_amp(model, device, optimizer, use_amp)

    if distributed:
        if APEX_PARALLEL_AVAILABLE:
            model = convert_syncbn_model(model)
            logger.info("Multi-GPU Training via DistributedDataParallel and apex.parallel")
        else:
            logger.info("Multi-GPU Training via DistributedDataParallel")

        # for some models DistributedDataParallel might complain about parameters
        # not contributing to loss. find_used_parameters remedies that.
        model = WrappedDDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    elif torch.cuda.device_count() > 1 and device.type == "cuda":
        model = WrappedDataParallel(model)
        logger.info("Multi-GPU Training via DataParallel")

    return model, optimizer


def _init_amp(model, device, optimizer=None, use_amp=None):
    model = model.to(device)
    if use_amp and optimizer:
        if AMP_AVAILABLE:
            model, optimizer = amp.initialize(model, optimizer, opt_level=use_amp)
        else:
            logger.warning(f"Can't find AMP although you specificed to use amp with level {use_amp}. Will continue without AMP ...")

    return model, optimizer
