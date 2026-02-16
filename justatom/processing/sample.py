import logging
from typing import Any

logger = logging.getLogger(__name__)


class Sample:
    """A single training/test sample. This should contain the input and the label. Is initialized with
    the human readable clear_text. Over the course of data preprocessing, this object is populated
    with tokenized and featurized versions of the data."""

    def __init__(
        self,
        id: str,
        clear_text: dict,
        tokenized: dict | None = None,
        features: dict[str, Any] | list[dict[str, Any]] | None = None,
    ):
        """
        :param id: The unique id of the sample
        :param clear_text: A dictionary containing various human readable fields (e.g. text, label).
        :param tokenized: A dictionary containing the tokenized version of clear text plus helpful meta data: offsets (start position of each token in the original text) and start_of_word (boolean if a token is the first one of a word).
        :param features: A dictionary containing features in a vectorized format needed by the model to process this sample.
        """  # noqa: E501
        self.id = id
        self.clear_text = clear_text
        self.features = features
        self.tokenized = tokenized

    def __str__(self):
        if self.clear_text:
            clear_text_str = "\n \t".join(
                [k + ": " + str(v) for k, v in self.clear_text.items()]
            )
            if len(clear_text_str) > 3000:
                clear_text_str = (
                    clear_text_str[:3_000]
                    + f"\nTHE REST IS TOO LONG TO DISPLAY. Remaining chars :{len(clear_text_str)-3_000}"
                )
        else:
            clear_text_str = "None"

        if self.features:
            if isinstance(self.features, list):  # noqa: SIM108
                features = self.features[0]
            else:
                features = self.features
            feature_str = "\n \t".join([k + ": " + str(v) for k, v in features.items()])
        else:
            feature_str = "None"

        if self.tokenized:
            tokenized_str = "\n \t".join(
                [k + ": " + str(v) for k, v in self.tokenized.items()]
            )
            if len(tokenized_str) > 3000:
                tokenized_str = (
                    tokenized_str[:3_000]
                    + f"\nTHE REST IS TOO LONG TO DISPLAY. Remaining chars: {len(tokenized_str)-3_000}"
                )
        else:
            tokenized_str = "None"
        s = (
            f"ID: {self.id}\n"
            f"Clear Text: \n \t{clear_text_str}\n"
            f"Tokenized: \n \t{tokenized_str}\n"
            f"Features: \n \t{feature_str}\n"
            "_____________________________________________________"
        )
        return s


class SampleBasket:
    """An object that contains one source text and the one or more samples that will be processed. This
    is needed for tasks like question answering where the source text can generate multiple input - label
    pairs."""

    def __init__(
        self,
        id_internal: int | str | None,
        raw: dict,
        id_external: str | None = None,
        samples: list[Sample] | None = None,
    ):
        """
        :param id_internal: A unique identifying id. Used for identification within patronum flow.
        :param external_id: Used for identification outside of patronum flow. E.g. if another framework wants to pass along its own id with the results.
        :param raw: Contains the various data needed to form a sample. It is ideally in human readable form.
        :param samples: An optional list of Samples used to populate the basket at initialization.
        """  # noqa: E501
        self.id_internal = id_internal
        self.id_external = id_external
        self.raw = raw
        self.samples = samples


__all__ = ["Sample", "SampleBasket"]
