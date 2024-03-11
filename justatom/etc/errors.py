"""Custom Errors"""

from typing import Optional


class IError(Exception):
    """
    This error wraps its source transparently in such a way that its attributes
    can be accessed directly: for example, if the original error has a `message` attribute,
    `HaystackError.message` will exist and have the expected content.
    If send_message_in_event is set to True (default), the message will be sent as part of a telemetry event reporting the error.
    The messages of errors that might contain user-specific information will not be sent, e.g., DocumentStoreError.
    """

    def __init__(
        self,
        message: Optional[str] = None,
        docs_link: Optional[str] = None,
        send_message_in_event: bool = True,
    ):
        super().__init__()
        if message:
            self.message = message
        self.docs_link = None

    def __getattr__(self, attr):
        # If self.__cause__ is None, it will raise the expected AttributeError
        getattr(self.__cause__, attr)

    def __str__(self):
        if self.docs_link:
            docs_message = f"\n\nCheck out the documentation at {self.docs_link}"
            return self.message + docs_message
        return self.message

    def __repr__(self):
        return str(self)


class ModelingError(IError):
    """Exception for issues raised by the modeling module"""

    def __init__(
        self,
        message: Optional[str] = None,
        docs_link: Optional[str] = "https://github.com/atomicai/patronum",
    ):
        super().__init__(message=message, docs_link=docs_link)


class PipelineError(IError):
    """Exception for issues raised within a pipeline"""

    def __init__(
        self,
        message: Optional[str] = None,
        docs_link: Optional[str] = "https://github.com/atomicai/patronum",
    ):
        super().__init__(message=message, docs_link=docs_link)


class PipelineSchemaError(PipelineError):
    """Exception for issues arising when reading/building the JSON schema of pipelines"""

    def __init__(self, message: Optional[str] = None):
        super().__init__(message=message)


class PipelineConfigError(PipelineError):
    """Exception for issues raised within a pipeline's config file"""

    def __init__(self, message: Optional[str] = None, docs_link: Optional[str] = None):
        super().__init__(message=message, docs_link=docs_link)


class DocumentStoreError(IError):
    """Exception for issues that occur in a document store"""

    def __init__(self, message: Optional[str] = None, send_message_in_event: bool = False):
        super().__init__(message=message, send_message_in_event=send_message_in_event)


class FilterError(DocumentStoreError):
    """Exception for issues that occur building complex filters"""

    def __init__(self, message: Optional[str] = None):
        super().__init__(message=message)


class DuplicateDocumentError(DocumentStoreError, ValueError):
    """Exception for Duplicate document"""

    def __init__(self, message: Optional[str] = None):
        super().__init__(message=message)


__all__ = ["FilterError", "DuplicateDocumentError", "DocumentStoreError", "ModelingError"]
