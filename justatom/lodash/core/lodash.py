import logging
import os
import random
import tarfile
import tempfile
import zipfile
from itertools import islice
from pathlib import Path
from typing import Any

import lib_programname
import numpy as np
import requests
import torch
from nicely import Printer
from tqdm import tqdm
from transformers import cached_path

from justatom.lodash.loader import MLoader

logger = logging.getLogger("-_- _ _ _ -_-")


class Lodash(MLoader):
    def __init__(self, max_seq=128, max_lines=18, indent=""):
        self.cout = Printer(max_seq=max_seq, max_lines=max_lines, indent=indent)

    def print(self, obj):
        self.cout.print(obj)

    @staticmethod
    def http_get(url, temp_file, proxies=None):
        """
        :param url:
        :param temp_file:
        :param proxies:
        :return:
        """
        req = requests.get(url, stream=True, proxies=proxies)
        content_length = req.headers.get("Content-Length")
        total = int(content_length) if content_length is not None else None
        progress = tqdm(unit="B", total=total)
        for chunk in req.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                progress.update(len(chunk))
                temp_file.write(chunk)
        progress.close()

    @staticmethod
    def fetch_archive_from_http(url, output_dir, proxies=None):
        """
        Fetch an archive (zip or tar.gz) from a url via http and extract content to an output directory.

        :param url: http address
        :type url: str
        :param output_dir: local path
        :type output_dir: str
        :param proxies: proxies details as required by requests library
        :type proxies: dict
        :return: bool if anything got fetched
        """
        # verify & prepare local directory
        path = Path(output_dir)
        if not path.exists():
            path.mkdir(parents=True)

        is_not_empty = len(list(Path(path).rglob("*"))) > 0
        if is_not_empty:
            logger.info(f"Found data stored in `{output_dir}`. Delete this first if you really want to fetch new data.")
            return False
        else:
            logger.info(f"Fetching from {url} to `{output_dir}`")

            # download & extract
            with tempfile.NamedTemporaryFile() as temp_file:
                Lodash.http_get(url, temp_file, proxies=proxies)
                temp_file.flush()
                temp_file.seek(0)  # making tempfile accessible
                # extract
                if url[-4:] == ".zip":
                    archive = zipfile.ZipFile(temp_file.name)
                    archive.extractall(output_dir)
                elif url[-7:] == ".tar.gz":
                    archive = tarfile.open(temp_file.name)
                    archive.extractall(output_dir)
                # temp_file gets deleted here
            return True

    @staticmethod
    def load_from_cache(pretrained_model_name_or_path, s3_dict, **kwargs):
        # Adjusted from HF Transformers to fit loading WordEmbeddings from deepsets s3
        # Load from URL or cache if already cached
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)

        s3_file = s3_dict[pretrained_model_name_or_path]
        try:
            resolved_file = cached_path(
                s3_file,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
            )

            if resolved_file is None:
                raise OSError

        except OSError:
            if pretrained_model_name_or_path in s3_dict:
                msg = f"Couldn't reach server at '{s3_file}' to download data."
            else:
                msg = (
                    f"Model name '{pretrained_model_name_or_path}' was not found in model name list. "
                    f"We assumed '{s3_file}' was a path, a model identifier, or url to a configuration file or "
                    "a directory containing such a file but couldn't find any such file at this path or url."
                )
            raise OSError(msg)  # noqa: B904

        if resolved_file == s3_file:
            logger.info(f"loading file {s3_file}")
        else:
            logger.info(f"loading file {s3_file} from cache at {resolved_file}")

        return resolved_file

    @staticmethod
    def get_batches_from_generator(iterable, n):
        """
        Batch elements of an iterable into fixed-length chunks or blocks.
        """
        it = iter(iterable)
        x = tuple(islice(it, n))
        while x:
            yield x
            x = tuple(islice(it, n))

    @staticmethod
    def is_in(obj: Any, method: str):
        return method in obj.__dict__

    @staticmethod
    def normalize(x, default):
        if x is None:
            return default
        return x

    @staticmethod
    def root() -> Path:
        return lib_programname.get_path_executed_script()

    @staticmethod
    def set_all_seeds(seed, deterministic_cudnn=False):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_cudnn:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    @staticmethod
    def fire_device_settings(use_cuda, rank=-1):
        if not use_cuda:
            device = torch.device("cpu")
            n_gpu = 0
        elif rank == -1:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if not torch.cuda.is_available():  # noqa: SIM108
                n_gpu = 0
            else:
                n_gpu = torch.cuda.device_count()
        else:
            device = torch.device("cuda", rank)
            torch.cuda.set_device(device)
            n_gpu = 1
            torch.distributed.init_process_group(backend="nccl")
        logger.info(f"Device: {str(device).upper()}")
        logger.info(f"GPU(s): {str(n_gpu)}")
        logger.info(f"Distributed Training: {bool(rank != -1)}")
