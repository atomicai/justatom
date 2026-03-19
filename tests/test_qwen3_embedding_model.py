from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from justatom.modeling.mask import ILanguageModel
from justatom.modeling.prime import Qwen3EmbeddingModel


class _FakeQwenModel(nn.Module):
    def __init__(self, hidden_size: int = 1024):
        super().__init__()
        self.hidden_size = hidden_size
        self.config = SimpleNamespace(hidden_size=hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        output_hidden_states: bool = False,
    ):
        bs, seq_len = input_ids.shape
        total = bs * seq_len * self.hidden_size
        last_hidden = torch.arange(total, dtype=torch.float32, device=input_ids.device).reshape(bs, seq_len, self.hidden_size) + 1.0
        hidden_states = None
        if output_hidden_states:
            hidden_states = (last_hidden * 0.5, last_hidden, last_hidden * 1.5)
        return SimpleNamespace(last_hidden_state=last_hidden, hidden_states=hidden_states)


class Qwen3EmbeddingModelTest(unittest.TestCase):
    def test_ilanguage_model_load_uses_qwen_mapping(self):
        with patch(
            "justatom.modeling.prime.AutoModel.from_pretrained",
            return_value=_FakeQwenModel(),
        ) as mocked:
            lm = ILanguageModel.load("Qwen/Qwen3-Embedding-0.6B")

        self.assertIsInstance(lm, Qwen3EmbeddingModel)
        mocked.assert_called_once_with("Qwen/Qwen3-Embedding-0.6B")

    def test_forward_pos_branch_with_target_dim(self):
        model = Qwen3EmbeddingModel(model_name_or_instance=_FakeQwenModel())
        input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
        attention_mask = torch.tensor([[0, 1, 1], [1, 1, 1]])
        pos_input_ids = torch.tensor([[7, 8, 9], [10, 11, 12]])
        pos_attention_mask = torch.tensor([[1, 1, 1], [0, 1, 1]])

        query_embs, pos_embs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pos_input_ids=pos_input_ids,
            pos_attention_mask=pos_attention_mask,
            norm=True,
            layer_idx=-1,
            target_dim=256,
        )

        self.assertEqual(query_embs.shape, (2, 256))
        self.assertEqual(pos_embs.shape, (2, 256))
        self.assertTrue(torch.allclose(query_embs.norm(dim=1), torch.ones(2), atol=1e-4))
        self.assertTrue(torch.allclose(pos_embs.norm(dim=1), torch.ones(2), atol=1e-4))

    def test_target_dim_validation(self):
        model = Qwen3EmbeddingModel(model_name_or_instance=_FakeQwenModel())
        input_ids = torch.tensor([[1, 2, 3]])
        attention_mask = torch.tensor([[1, 1, 1]])

        with self.assertRaises(ValueError):
            model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                target_dim=16,
            )


if __name__ == "__main__":
    unittest.main()
