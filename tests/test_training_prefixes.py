from justatom.api.train import resolve_train_kwargs
from justatom.processing.prime import TrainWithContrastiveProcessor


def test_training_prefixes_are_forwarded_to_the_training_job():
    query_prefix = "Instruct: retrieve relevant passages\nQuery:"
    kwargs = resolve_train_kwargs(
        config={
            "dataset": {"name_or_path": "justatom"},
            "training": {
                "query_prefix": query_prefix,
                "content_prefix": "",
                "max_query_seq_len": 128,
            },
        }
    )

    assert kwargs["query_prefix"] == query_prefix
    assert kwargs["content_prefix"] == ""
    assert kwargs["max_query_seq_len"] == 128


class _RecordingTokenizer:
    def __init__(self):
        self.max_lengths = []

    def __call__(self, texts, *, truncation, max_length, padding):
        self.max_lengths.append(max_length)
        return {
            "input_ids": [[1] * max_length for _ in texts],
            "attention_mask": [[1] * max_length for _ in texts],
        }


def test_contrastive_processor_uses_separate_query_and_document_limits():
    tokenizer = _RecordingTokenizer()
    processor = TrainWithContrastiveProcessor(
        tokenizer=tokenizer,
        max_seq_len=512,
        max_query_seq_len=128,
    )

    processor.dataset_from_dicts([{"query": "q", "content": "p"}])

    assert tokenizer.max_lengths == [128, 512]
