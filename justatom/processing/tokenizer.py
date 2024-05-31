from transformers import AutoTokenizer, PreTrainedTokenizer
from typing import Optional, Union, Dict
from loguru import logger
import simplejson as json
from typing import Tuple
from justatom.tooling.stl import merge_in_order
from pathlib import Path


class ITokenizer(PreTrainedTokenizer):

    subclasses = {}

    def __init_subclass__(cls, **kwargs):
        """This automatically keeps track of all available subclasses.
        Enables generic load() and load_from_dir() for all specific Processor implementation.
        """
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls

    def __init__(self, max_len, pad_token, **props):
        super().__init__(max_len=max_len, pad_token=pad_token)

    @classmethod
    def from_pretrained(cls, where) -> PreTrainedTokenizer:
        # Here we do check if `where` is a directory or `huggingface` tokenizer-name
        where_path = Path(where)
        if where_path.is_dir():
            config_path = where_path / "tokenizer_config.json"
            assert config_path.exists(), f"[tokenizer_config.json] is missing in [{str(where)}] directory"
            with open(config_path) as fp:
                config = json.load(fp)
            klass = config["tokenizer_class"]
            if klass in cls.subclasses:
                return cls.subclasses[klass].load(config=config, where=where)
            else:
                return AutoTokenizer.from_pretrained(where)
        else:
            # try to ignite huggingface `transformers` tokenizer
            try:
                tokenizer = ignite_hf_tokenizer(where)
            except:
                msg = f"The provided name [{where}] neither directory nor recognized tokenizer name from `huggingface.co`"
                logger.error(msg)
                raise ValueError(msg)
            else:
                return tokenizer


class WHITESPACETokenizer(ITokenizer):
    def __init__(self, vocab: Dict[str, int], max_len: int = None, pad_token: str = None, **props):
        self.__token_ids = vocab
        self.__id_tokens: Dict[int, str] = {value: key for key, value in vocab.items()}
        self.pad_token = pad_token
        super().__init__(max_len=max_len, pad_token=pad_token, **props)

    @classmethod
    def load(cls, config, where):
        vocab = ignite_vocab_tokens(where)
        tokens = ignite_special_tokens(where)
        props = merge_in_order(tokens, config)
        return cls(vocab=vocab, **props)

    def save(self, where):
        self.save_pretrained(where)

    def _tokenize(self, text: str, **kwargs):
        return text.split(" ")

    def _convert_token_to_id(self, token: str) -> int:
        return self.__token_ids[token] if token in self.__token_ids else self.unk_token_id

    def _convert_id_to_token(self, index: int) -> str:
        return self.__id_tokens[index] if index in self.__id_tokens else self.unk_token

    def get_vocab(self) -> Dict[str, int]:
        return self.__token_ids.copy()

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if filename_prefix is None:
            filename_prefix = ""
        vocab_path = Path(save_directory, filename_prefix + "vocab.json")
        with open(vocab_path, "w") as fp:
            json.dump(self.__token_ids, fp)
        return (str(vocab_path),)

    @property
    def vocab_size(self) -> int:
        return len(self.__token_ids)


def ignite_hf_tokenizer(
    pretrained_model_name_or_path: str,
    revision: str = None,
    use_fast: bool = True,
    use_auth_token: Optional[Union[str, bool]] = None,
    **kwargs,
) -> PreTrainedTokenizer:

    model_name_or_path = str(pretrained_model_name_or_path)
    params = {}
    if any(tokenizer_type in model_name_or_path for tokenizer_type in ["albert", "xlnet"]):
        params["keep_accents"] = True

    return AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path,
        revision=revision,
        use_fast=use_fast,
        use_auth_token=use_auth_token,
        **params,
        **kwargs,
    )


def ignite_vocab_tokens(where) -> Dict[str, int]:
    fp = Path(where) / "vocab.json"
    if not fp.exists():
        msg = f"The vocab file {str(fp)} doesn't exist. Did you save it by calling `save_pretrained(...)` ?"
        logger.error(msg)
        raise ValueError(msg)
    else:
        with open(fp) as fin:
            vocab = json.load(fin)
    return vocab


def ignite_special_tokens(where) -> Dict[str, str]:
    fp = Path(where) / "special_tokens_map.json"
    if not fp.exists():
        msg = f"The special tokens file [{str(fp)}] doesn't exist. Did you save it by calling `save_pretrained(...)` ?"
        logger.error(msg)
        raise ValueError(msg)
    else:
        with open(fp) as fin:
            config_tokens = json.load(fin)
        tokens = {k: v["content"] for k, v in config_tokens.items()}
    return tokens


__all__ = ["ITokenizer", "WHITESPACETokenizer"]
