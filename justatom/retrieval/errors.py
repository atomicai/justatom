class RetrievalError(Exception):
    pass


class ConfigurationError(RetrievalError, ValueError):
    pass


class EmbeddingError(RetrievalError):
    pass


class EmbeddingBackendError(EmbeddingError):
    pass


class EmbeddingResponseError(EmbeddingError):
    pass
