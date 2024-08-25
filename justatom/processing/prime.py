import torch
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from justatom.processing.mask import IProcessor
from justatom.processing.sample import Sample, SampleBasket
from justatom.processing.tokenizer import ITokenizer


class INFERProcessor(IProcessor):
    """
    (1) This type of processor is responsible for fast inference using `tokenizers` custom implementation.
    (2) It performs only necessary transformation and avoids typical pre-processing bottleneck such as `regex` use.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        max_seq_len: int = 512,
        do_lower_case: bool = False,
        content_field: str = "content",
        prefix_field: str = "prefix",
        prefix: str = "",
    ):
        super(INFERProcessor, self).__init__()  # noqa: UP008
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.do_lower_case = do_lower_case
        self.content_field = content_field
        self.prefix_field = prefix_field
        self.prefix = prefix

    def dataset_from_dicts(self, dicts, indices=None, return_baskets=False):
        if indices is None:
            indices = []
        baskets = []
        docs = [
            self.do_prefix(
                x.get(self.content_field),
                pref=x.get("meta", {}).get(self.prefix_field, self.prefix),
            )
            for x in dicts
        ]

        tokenized_batch = self.tokenizer(docs, truncation=True, max_length=self.max_seq_len, padding="max_length")

        input_ids_batch = tokenized_batch["input_ids"]
        atten_ids_batch = tokenized_batch["attention_mask"]

        for sample, input_ids, att_ids in zip(docs, input_ids_batch, atten_ids_batch, strict=False):
            tokenized = {}
            features = dict(input_ids=input_ids, attention_mask=att_ids)

            cur_sample = Sample(id="", clear_text=sample, tokenized=tokenized, features=[features])
            cur_basket = SampleBasket(id_internal=None, raw=sample, id_external=None, samples=[cur_sample])

            baskets.append(cur_basket)

        problematic_ids = set()
        dataset, tensornames = self._create_dataset(baskets)

        if return_baskets:
            return dataset, tensornames, problematic_ids, baskets
        else:
            return dataset, tensornames, problematic_ids


class M1Processor(IProcessor):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        max_seq_len: int = 512,
        do_lower_case: bool = False,
        content_field: str = "content",
        prefix_field: str = "prefix",
        prefix: str = "",
    ):
        super(M1Processor, self).__init__()  # noqa: UP008
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.do_lower_case = do_lower_case
        self.content_field = content_field
        self.prefix_field = prefix_field

    def dataset_from_dicts(self, dicts, indices=None, return_baskets=False):
        if indices is None:
            indices = []
        baskets = []
        docs = [
            self.do_prefix(
                x.get(self.content_field),
                pref=x.get("meta", {}).get(self.prefix_field, self.prefix),
            )
            for x in dicts
        ]

        tokenized_batch = self.tokenizer(docs, truncation=True, max_length=self.max_seq_len, padding="max_length")

        input_ids_batch = tokenized_batch["input_ids"]
        atten_ids_batch = tokenized_batch["attention_mask"]

        for sample, input_ids, att_ids in zip(docs, input_ids_batch, atten_ids_batch, strict=False):
            tokenized = {}
            features = dict(input_ids=input_ids, attention_mask=att_ids)

            cur_sample = Sample(id="", clear_text=sample, tokenized=tokenized, features=[features])
            cur_basket = SampleBasket(id_internal=None, raw=sample, id_external=None, samples=[cur_sample])

            baskets.append(cur_basket)

        problematic_ids = set()
        dataset, tensornames = self._create_dataset(baskets)

        if return_baskets:
            return dataset, tensornames, problematic_ids, baskets
        else:
            return dataset, tensornames, problematic_ids


class TripletProcessor(IProcessor):
    """
    TRIplet Language Model separation for different encoder(s) LM processor that performs grouping for similarity fine-tuning.
    Preprocess samples for `justatom.training.loss.TripletLoss` loss function
    for separating samples at least by `margin` distance using both `negative`, `positive` samples.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        max_seq_len: int = 512,
        prefix: str = "",
        **kwargs,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.prefix = prefix

    @classmethod
    def load(cls, where, config: dict, **props):
        tokenizer = ITokenizer.from_pretrained(where)
        return cls(tokenizer=tokenizer, **config)

    def dataset_from_dicts(self, dicts, indices=None, return_baskets=False, debug=False):
        if indices is None:
            indices = []
        baskets = []
        docs = [self.do_prefix(x["content"], pref=x.get("meta", {}).get("prefix", self.prefix)) for x in dicts]
        groups = [hash(x.get("meta", {}).get("group", 0)) for x in dicts]
        # ---
        tokenized_batch = self.tokenizer(docs, truncation=True, max_length=self.max_seq_len, padding="max_length")
        # ---
        input_ids_batch = tokenized_batch["input_ids"]
        atten_ids_batch = tokenized_batch["attention_mask"]
        # ---
        for sample, input_ids, att_ids, group_ids in zip(docs, input_ids_batch, atten_ids_batch, groups, strict=False):
            tokenized = {}
            # TODO: Convert `group_id` to compatable format
            features = dict(
                input_ids=input_ids,
                attention_mask=att_ids,
                group_ids=torch.tensor(group_ids),
            )

            cur_sample = Sample(id="", clear_text=sample, tokenized=tokenized, features=[features])
            cur_basket = SampleBasket(id_internal=None, raw=sample, id_external=None, samples=[cur_sample])
            baskets.append(cur_basket)

        problematic_ids = set()
        dataset, tensornames = self._create_dataset(baskets)

        if return_baskets:
            return dataset, tensornames, problematic_ids, baskets
        else:
            return dataset, tensornames, problematic_ids


class ContrastiveProcessor(IProcessor):
    """
    ContrastiveProcessor separation for different encoder(s) LM processor that performs grouping for similarity fine-tuning.
    Preprocess samples for `justatom.training.loss.ContrastiveLoss` loss function
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        max_seq_len: int = 512,
        queries_prefix: str = "query:",
        queries_field: str = "query",
        pos_queries_field: str = "content",
        pos_queries_prefix: str = "passage:",
        **kwargs,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.queries_field = queries_field
        self.queries_prefix = queries_prefix
        self.pos_queries_prefix = pos_queries_prefix
        self.pos_queries_field = pos_queries_field

    @classmethod
    def load(cls, where, config: dict, **props):
        tokenizer = ITokenizer.from_pretrained(where)
        return cls(tokenizer=tokenizer, **config)

    def dataset_from_dicts(self, dicts, indices=None, return_baskets=False, debug=False):
        if indices is None:
            indices = []
        baskets = []
        queries = [
            self.do_prefix(
                x.get(self.queries_field),  # query
                pref=x.get("meta", {}).get("queries_prefix", self.queries_prefix),
            )
            for x in dicts
        ]
        pos_queries = [
            self.do_prefix(
                x.get(self.pos_queries_field),  # content
                pref=x.get("meta", {}).get("pos_queries_prefix", self.pos_queries_prefix),
            )
            for x in dicts
        ]
        # ---
        tokenized_queries_batch = self.tokenizer(queries, truncation=True, max_length=self.max_seq_len, padding="max_length")
        tokenized_pos_queries_batch = self.tokenizer(
            pos_queries,
            truncation=True,
            max_length=self.max_seq_len,
            padding="max_length",
        )
        # ---
        queries_input_ids_batch = tokenized_queries_batch["input_ids"]
        pos_queries_input_ids_batch = tokenized_pos_queries_batch["input_ids"]
        queries_atten_ids_batch = tokenized_queries_batch["attention_mask"]
        pos_queries_atten_ids_batch = tokenized_pos_queries_batch["attention_mask"]

        for (
            sample,
            pos_sample,
            queries_input_ids,
            queries_att_ids,
            pos_queries_input_ids,
            pos_queries_att_ids,
        ) in zip(
            queries,
            pos_queries,
            queries_input_ids_batch,
            queries_atten_ids_batch,
            pos_queries_input_ids_batch,
            pos_queries_atten_ids_batch,
            strict=False,
        ):
            features = dict(
                input_ids=queries_input_ids,
                attention_mask=queries_att_ids,
                pos_input_ids=pos_queries_input_ids,
                pos_attention_mask=pos_queries_att_ids,
            )

            cur_sample = Sample(
                id="",
                clear_text=[sample, pos_sample],
                tokenized={},
                features=[features],
            )
            cur_basket = SampleBasket(id_internal=None, raw=sample, id_external=None, samples=[cur_sample])
            baskets.append(cur_basket)

        problematic_ids = set()
        dataset, tensornames = self._create_dataset(baskets)

        if return_baskets:
            return dataset, tensornames, problematic_ids, baskets
        else:
            return dataset, tensornames, problematic_ids


class M2Processor(IProcessor):
    """
    (1) This processor uses two different tokenizers. First one transforms `queries` while the second one - `passages`.
    (2) It DOES NOT expect queries to have labels.
    """

    def __init__(
        self,
        query_tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,  # type: ignore
        passage_tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,  # type: ignore
        do_lower_case: bool = False,
        max_seq_len_query: int = 512,
        max_seq_len_passage: int = 512,
        data_dir: str = "",
        metric=None,  # type: ignore
        dev_split: float = 0.1,
        proxies: dict | None = None,
        max_samples: int | None = None,
        embed_title: bool = True,
        num_positives: int = 1,
        num_hard_negatives: int = 1,
        shuffle_negatives: bool = True,
        shuffle_positives: bool = False,
        label_list: list[str] | None = None,
        **kwargs,
    ):
        """
        :param query_tokenizer: Used to split a question (str) into tokens
        :param passage_tokenizer: Used to split a passage (str) into tokens.
        :param max_seq_len_query: Query samples are truncated after this many tokens.
        :param max_seq_len_passage: Context/Passage Samples are truncated after this many tokens.
        :param data_dir: The directory in which the train and dev files can be found.
                         If not available the dataset will be loaded automatically
                         if the last directory has the same name as a predefined dataset.
                         These predefined datasets are defined as the keys in the dict at
                         `haystack.basics.data_handler.utils.DOWNSTREAM_TASK_MAP <https://github.com/deepset-ai/haystack/blob/main/haystack/basics/data_handler/utils.py>`_.
        :param metric: name of metric that shall be used for evaluation, e.g. "acc" or "f1_macro".
                 Alternatively you can also supply a custom function, that takes preds and labels as args and returns a numerical value.
                 For using multiple metrics supply them as a list, e.g ["acc", my_custom_metric_fn].
        :param proxies: proxy configuration to allow downloads of remote datasets.
                        Format as in  "requests" library: https://2.python-requests.org//en/latest/user/advanced/#proxies
        :param max_samples: maximum number of samples to use
        :param embed_title: Whether to embed title in passages during tensorization (bool),
        :param num_hard_negatives: maximum number to hard negative context passages in a sample
        :param num_positives: maximum number to positive context passages in a sample
        :param shuffle_negatives: Whether to shuffle all the hard_negative passages before selecting the num_hard_negative number of passages
        :param shuffle_positives: Whether to shuffle all the positive passages before selecting the num_positive number of passages
        :param label_list: list of labels to predict. Usually ["hard_negative", "positive"]
        :param kwargs: placeholder for passing generic parameters
        """  # noqa: E501
        super(M2Processor, self).__init__()  # noqa: UP008
        if metric:
            pass

    def dataset_from_dicts(self, dicts):
        pass


class ATOMICProcessor(IProcessor):
    """
    (1) This processor uses two different tokenizers. First one transforms `queries` while the second one - `passages`.
    (2) It DOES expect each `query` to have its own `label`.
    """

    def __init__(
        self,
        query_tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,  # type: ignore
        passage_tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,  # type: ignore
        do_lower_case_query: bool = False,
        do_lower_case_passage: bool = False,
        max_seq_len_query: int = 128,
        max_seq_len_passage: int = 512,
        data_dir: str = "",
        metric=None,  # type: ignore
        dev_split: float = 0.1,
        proxies: dict | None = None,
        max_samples: int | None = None,
        embed_title: bool = True,
        num_positives: int = 1,
        num_hard_negatives: int = 1,
        shuffle_negatives: bool = True,
        shuffle_positives: bool = False,
        label_list: list[str] | None = None,
        label_queries_list: list[str] | None = None,
    ):
        super(ATOMICProcessor, self).__init__()  # noqa: UP008

    def dataset_from_dicts(self, dicts):
        pass


__all__ = ["TripletProcessor", "ContrastiveProcessor", "IProcessor"]
