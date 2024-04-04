import torch.nn as nn
import torch
import simplejson as json
import torch.functional as F
from pathlib import Path
import inspect
import copy
from typing import List, Union, Optional, Dict, Any


def _get_clones(module, N):
    # FIXME: copy.deepcopy() is not defined on nn.module
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Conv1DBlock(nn.Module):
    """
    A Convolutional Block module for transforming N x a onto N x b tensor. Designed to work with textual embeddings (not CV).
    """

    def __init__(self, kernel_size):
        super().__init__()


class ResBlock(nn.Module):
    """
    A Residual Block module.

    This module performs a linear transformation followed by a SiLU activation,
    and then adds the result to the original input, creating a residual connection.

    Args:
        hidden_size (int): The size of the hidden layers in the block.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        # Initialize as an identity mapping
        nn.init.zeros_(self.linear.weight)
        # Use SiLU activation to keep consistent with the Llama model
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Forward pass of the ResBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output after the residual connection and activation.
        """
        return x + self.act(self.linear(x))


class FFBlock(nn.Module):
    """
    Feed Forward Block module.
    """

    def __init__(self, layer_dims: List[int]):
        # Todo: Consider having just one input argument
        super(FFBlock, self).__init__()
        self.layer_dims = layer_dims
        # If read from config the input will be string
        L = len(layer_dims) - 1
        layers_all = []
        # TODO: IS this needed?
        self.output_size = layer_dims[-1]

        for i in range(L):
            size_in = layer_dims[i]
            size_out = layer_dims[i + 1]
            layer = nn.Linear(size_in, size_out)
            layers_all.append(layer)
        self.feed_forward = nn.Sequential(*layers_all)

    def forward(self, X):
        logits = self.feed_forward(X)
        return logits


class IEmbedding(nn.Module):
    props = ("embedding_dim", "vocab_size")

    def __init__(self, embedding_dim, vocab_size):
        super(IEmbedding, self).__init__()
        self.config = dict(embedding_dim=embedding_dim, vocab_size=vocab_size)
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim=embedding_dim)

        self.norm = nn.LayerNorm(self.embedding_dim)

    @property
    def size(self):
        return self.embedding.size

    @classmethod
    def load(cls, pretrained_model_name_or_path, **kwargs):
        _path = Path(pretrained_model_name_or_path) / "embedding"
        assert _path.exists(), f"Path doesn't exists. Please double-check the path {str(pretrained_model_name_or_path)}"
        farm_config = Path(_path) / "embedding_config.json"
        farm_model = Path(_path) / "embedding_model.bin"
        _model = None
        if farm_config.exists():
            with open(str(farm_config), "r") as fin:
                config = json.load(fin)
            # vocab_size, embedding_dim = config["vocab_size"], config["embedding_dim"]
            props = {k: config[k] for k in inspect.signature(cls.__init__).parameters.keys() if k in cls.props}
            _model = cls(**props)
            _model.load_state_dict(torch.load(str(farm_model)))
        else:
            vocab_size = 5218
            embedding_dim: int = 64
            _model = cls(vocab_size=vocab_size, embedding_dim=embedding_dim)

        return _model

    def save_config(self, save_dir: Union[Path, str]):
        save_filename = Path(save_dir) / "embedding_config.json"
        config = copy.deepcopy(self.config)
        config["klass"] = self.__class__.__name__
        with open(str(save_filename), "w") as f:
            f.write(json.dumps(config))

    def save(self, save_dir: Union[str, Path], state_dict: Optional[Dict[Any, Any]] = None):
        """
        Save the model `state_dict` and its configuration file so that it can be loaded again.

        :param save_dir: The directory in which the model should be saved.
        :param state_dict: A dictionary containing the whole state of the module, including names of layers. By default, the unchanged state dictionary of the module is used.
        """
        save_dir = Path(save_dir) / "embedding"
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        save_name = Path(save_dir) / "embedding_model.bin"
        if not state_dict:
            state_dict = self.state_dict()
        torch.save(state_dict, save_name)
        self.save_config(save_dir)

    def forward(self, input_tensor):
        sentence_size = input_tensor.size(-1)
        pos_tensor = self.attention_position(self.embedding_dim, input_tensor)

        segment_tensor = torch.zeros_like(input_tensor).to(input_tensor.device)
        segment_tensor[:, sentence_size // 2 + 1 :] = 1

        output = self.embedding(input_tensor) + pos_tensor
        return self.norm(output)

    def attention_position(self, dim, input_tensor):
        batch_size = input_tensor.size(0)
        sentence_size = input_tensor.size(-1)

        pos = torch.zeros(sentence_size, dtype=torch.long).to(input_tensor.device)
        d = torch.arange(dim, dtype=torch.long).to(input_tensor.device)
        d = 2 * d / dim

        pos = pos.unsqueeze(1)
        pos = pos / (1e4**d)

        # pos[:, ::2] = torch.sin(pos[:, ::2])
        # pos[:, 1::2] = torch.cos(pos[:, 1::2])

        return pos.expand(batch_size, *pos.size())

    def numeric_position(self, dim, input_tensor):
        pos_tensor = torch.arange(dim, dtype=torch.long).to(input_tensor.device)
        return pos_tensor.expand_as(input_tensor)


class AttentionHead(nn.Module):
    def __init__(self, dim_inp, dim_out):
        super(AttentionHead, self).__init__()

        self.dim_inp = dim_inp

        self.q = nn.Linear(dim_inp, dim_out)
        self.k = nn.Linear(dim_inp, dim_out)
        self.v = nn.Linear(dim_inp, dim_out)

    def forward(self, input_tensor: torch.Tensor, attention_mask: torch.Tensor = None):
        query, key, value = self.q(input_tensor), self.k(input_tensor), self.v(input_tensor)

        scale = query.size(1) ** 0.5
        scores = torch.bmm(query, key.transpose(1, 2)) / scale  # batch_size x max_seq_len  x max_seq_len
        # TODO:
        # Чтобы веса аттеншена не распределялись на паддинг - надо эти паддинги сделать нулевыми (перед софтмаксом).
        # Но attention_mask приходит в нативной форме. batch_size x max_seq_len.
        # Поэтому на пред-софтмаксовые скоры мы можем просто сделать так:
        _scores = scores.masked_fill(attention_mask[:, None, :] == 0, -1e9)
        attn = F.softmax(_scores, dim=-1)
        context = torch.bmm(attn, value)

        return context


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, dim_inp, dim_out):
        super(MultiHeadAttention, self).__init__()

        self.heads = nn.ModuleList([AttentionHead(dim_inp, dim_out) for _ in range(num_heads)])
        self.linear = nn.Linear(dim_out * num_heads, dim_inp)
        self.norm = nn.LayerNorm(dim_inp)

    def forward(self, input_tensor: torch.Tensor, attention_mask: torch.Tensor):
        s = [head(input_tensor, attention_mask) for head in self.heads]
        scores = torch.cat(s, dim=-1)
        scores = self.linear(scores)
        return self.norm(scores)


class IAttention(nn.Module):
    props = ("embedding_dim", "dim_out", "attention_heads", "dropout")

    def __init__(self, embedding_dim, dim_out, attention_heads=4, dropout=0.1, **kwargs):
        super(IAttention, self).__init__()
        self.config = dict(
            attention_heads=attention_heads,
            embedding_dim=embedding_dim,
            dim_out=dim_out,
            dropout=dropout,
        )

        self.attention = MultiHeadAttention(
            attention_heads, embedding_dim, dim_out
        )  # batch_size x max_seq_len x embedding_size
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, dim_out),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(dim_out, embedding_dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(embedding_dim)

    @classmethod
    def load(cls, pretrained_model_name_or_path):
        _path = Path(pretrained_model_name_or_path) / "attention"
        assert _path.exists(), f"Path doesn't exists. Please double-check the path {str(pretrained_model_name_or_path)}"
        farm_config = Path(_path) / "attention_config.json"
        farm_model = Path(_path) / "attention_model.bin"
        if farm_config.exists():
            with open(str(farm_config), "r") as fin:
                config = json.load(fin)
            # drop the "klass" key and all of the rest keys not corresponding to constructor
            props = {k: config[k] for k in inspect.signature(cls.__init__).parameters.keys() if k in cls.props}
            _model = cls(**props)
            _model.load_state_dict(torch.load(str(farm_model)))
        else:
            embedding_dim = 64
            dim_out = 36
            _model = cls(embedding_dim=embedding_dim, dim_out=dim_out)
        return _model

    def save_config(self, save_dir: Union[Path, str]):
        save_filename = Path(save_dir) / "attention_config.json"
        config = copy.deepcopy(self.config)
        config["klass"] = self.__class__.__name__
        with open(str(save_filename), "w") as f:
            f.write(json.dumps(config))

    def save(self, save_dir: Union[str, Path], state_dict: Optional[Dict[Any, Any]] = None):
        """
        Save the model `state_dict` and its configuration file so that it can be loaded again.

        :param save_dir: The directory in which the model should be saved.
        :param state_dict: A dictionary containing the whole state of the module, including names of layers. By default, the unchanged state dictionary of the module is used.
        """
        save_dir = Path(save_dir) / "attention"
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        save_name = Path(save_dir) / "attention_model.bin"
        if not state_dict:
            state_dict = self.state_dict()
        torch.save(state_dict, save_name)
        self.save_config(save_dir)

    def forward(self, input_tensor: torch.Tensor, attention_mask: torch.Tensor):
        context = self.attention(input_tensor, attention_mask)
        res = self.feed_forward(context)
        return self.norm(res)


class MLAttention(nn.Module):
    props = ("embedding_dim", "dim_out", "attention_heads", "dropout", "num_blocks")

    def __init__(self, embedding_dim, dim_out, attention_heads=4, dropout=0.1, num_blocks=1, **kwargs):
        super(MLAttention, self).__init__()
        self.blocks = _get_clones(
            IAttention(embedding_dim, dim_out, attention_heads=attention_heads, dropout=dropout), num_blocks
        )
        self.config = dict(
            attention_heads=attention_heads,
            embedding_dim=embedding_dim,
            dim_out=dim_out,
            dropout=dropout,
            num_blocks=num_blocks,
        )

    def forward(self, input_tensor: torch.Tensor, attention_mask: torch.Tensor):
        out = input_tensor
        for m in self.blocks:
            response = m(out, attention_mask)
        return response

    @classmethod
    def load(cls, pretrained_model_name_or_path):
        _path = Path(pretrained_model_name_or_path) / "attention"
        assert _path.exists(), f"Path doesn't exists. Please double-check the path {str(pretrained_model_name_or_path)}"
        farm_config = Path(_path) / "mlattention_config.json"
        farm_model = Path(_path) / "mlattention_model.bin"
        if farm_config.exists():
            with open(str(farm_config), "r") as fin:
                config = json.load(fin)
            # drop the "klass" key and all of the rest keys not corresponding to constructor
            props = {k: config[k] for k in inspect.signature(cls.__init__).parameters.keys() if k in cls.props}
            _model = cls(**props)
            _model.load_state_dict(torch.load(str(farm_model)))
        else:
            embedding_dim = 64
            dim_out = 36
            _model = cls(embedding_dim=embedding_dim, dim_out=dim_out)
        return _model

    def save(self, save_dir: Union[str, Path], state_dict: Optional[Dict[Any, Any]] = None):
        """
        Save the model `state_dict` and its configuration file so that it can be loaded again.

        :param save_dir: The directory in which the model should be saved.
        :param state_dict: A dictionary containing the whole state of the module, including names of layers. By default, the unchanged state dictionary of the module is used.
        """
        save_dir = Path(save_dir) / "attention"
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        save_name = Path(save_dir) / "mlattention_model.bin"
        if not state_dict:
            state_dict = self.state_dict()
        torch.save(state_dict, save_name)
        self.save_config(save_dir)

    def save_config(self, save_dir: Union[Path, str]):
        save_filename = Path(save_dir) / "mlattention_config.json"
        config = copy.deepcopy(self.config)
        config["klass"] = self.__class__.__name__
        with open(str(save_filename), "w") as f:
            f.write(json.dumps(config))


__all__ = ["ResBlock", "FFBlock"]
