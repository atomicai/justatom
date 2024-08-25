from enum import Enum


class DuplicatePolicy(Enum):
    NONE = "none"
    SKIP = "skip"
    OVERWRITE = "overwrite"
    FAIL = "fail"


class SearchPolicy(Enum):
    BM25 = "bm25"


class LineSmoothChoice(Enum):
    NONE = "none"
    MAVG = "moving_average"
    LOESS = "loess"
    POLY = "poly"
