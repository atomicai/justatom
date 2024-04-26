from enum import Enum


class DuplicatePolicy(Enum):
    NONE = "none"
    SKIP = "skip"
    OVERWRITE = "overwrite"
    FAIL = "fail"


class SearchPolicy(Enum):

    BM25 = "bm25"
