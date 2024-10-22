import tempfile
import unittest

from transformers import AutoTokenizer

from justatom.modeling.head import ANNHead, LLHead
from justatom.modeling.prime import E5SModel
from justatom.processing.prime import INFERProcessor
from justatom.running.m1 import M1LMRunner
from justatom.running.mask import IMODELRunner


class M1LMRunnerTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small")
        processor = INFERProcessor(self.tokenizer)
        model = E5SModel()
        # ann_head = ANNHead()
        ll_head = LLHead(in_features=384, out_features=50)

        self.runner = M1LMRunner(
            model=model,
            prediction_heads=[ll_head],
            processor=processor,
        )

    def test_save(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            self.runner.save(tmpdirname)
            runner2 = IMODELRunner.load(tmpdirname)

        self.assertEqual(type(self.runner), type(runner2))
        for idx in range(len(self.runner.prediction_heads)):
            self.assertEqual(type(self.runner.prediction_heads[idx]), type(runner2.prediction_heads[idx]))

    def test_run(self):
        batch = self.tokenizer(["test text"], return_tensors="pt")
        out = self.runner(batch)[0]
        self.assertEqual(out.shape[-1], 50)


if __name__ == "__main__":
    unittest.main()
