import torch
from loguru import logger


def initialize_device_settings(use_gpus, local_rank=-1, use_amp=None):
    if not use_gpus:
        device = torch.device("cpu")
        n_gpu = 0
    elif local_rank == -1:
        if torch.cuda.is_available():
            device, n_gpu = torch.device("cuda"), torch.cuda.device_count()
        elif torch.backends.mps.is_available():
            device, n_gpu = torch.device("mps"), 1
        else:
            device, n_gpu = "cpu", 0
    else:
        if torch.backends.mps.is_available():
            device, n_gpu = "mps", 1
        elif not torch.cuda.is_available():
            msg = f"You specified [local_rank={local_rank}] but CUDA is not available."
            logger.error(msg)
            raise ValueError(msg)
        else:
            device = torch.device("cuda", local_rank)
            torch.cuda.set_device(device)
            n_gpu = 1
            # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.distributed.init_process_group(backend="nccl")
    logger.info(f"Using device: {str(device).upper()} ")
    logger.info(f"Number of GPUs: {n_gpu}")
    logger.info(f"Distributed Training: {bool(local_rank != -1)}")
    logger.info(f"Automatic Mixed Precision: {use_amp}")
    return device, n_gpu
