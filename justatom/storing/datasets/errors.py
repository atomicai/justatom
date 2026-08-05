class DatasetError(Exception):
    """Base error for dataset source resolution and reading."""


class DatasetNotFoundError(DatasetError, FileNotFoundError):
    """Raised when a source is neither a file nor a recognized remote ID."""


class UnsupportedDatasetSourceError(DatasetError, ValueError):
    """Raised when the source transport or syntax is unsupported."""


class UnsupportedDatasetFormatError(DatasetError, ValueError):
    """Raised when a local file has an unsupported format."""


class DatasetStreamingUnsupportedError(UnsupportedDatasetFormatError):
    """Raised when a format cannot honor the lazy iterator contract."""


class DatasetReadError(DatasetError):
    """Raised when a recognized dataset source cannot be read."""


__all__ = [
    "DatasetError",
    "DatasetNotFoundError",
    "DatasetReadError",
    "DatasetStreamingUnsupportedError",
    "UnsupportedDatasetFormatError",
    "UnsupportedDatasetSourceError",
]
