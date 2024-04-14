from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from enum import Enum

from justatom.etc.errors import DeserializationError

from weaviate.auth import AuthApiKey as WeaviateAuthApiKey
from weaviate.auth import AuthBearerToken as WeaviateAuthBearerToken
from weaviate.auth import AuthClientCredentials as WeaviateAuthClientCredentials
from weaviate.auth import AuthClientPassword as WeaviateAuthClientPassword

import os
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union, Type


class SecretType(Enum):
    TOKEN = "token"
    ENV_VAR = "env_var"

    def __str__(self):
        return self.value

    @staticmethod
    def from_str(string: str) -> "SecretType":
        map = {e.value: e for e in SecretType}
        type = map.get(string)
        if type is None:
            raise ValueError(f"Unknown secret type '{string}'")
        return type


class Secret(ABC):
    """
    Encapsulates a secret used for authentication.

    Usage example:
    ```python
    from haystack.components.generators import OpenAIGenerator
    from haystack.utils import Secret

    generator = OpenAIGenerator(api_key=Secret.from_token("<here_goes_your_token>"))
    ```
    """

    @staticmethod
    def from_token(token: str) -> "Secret":
        """
        Create a token-based secret. Cannot be serialized.

        :param token:
            The token to use for authentication.
        """
        return TokenSecret(_token=token)

    @staticmethod
    def from_env_var(env_vars: Union[str, List[str]], *, strict: bool = True) -> "Secret":
        """
        Create an environment variable-based secret. Accepts
        one or more environment variables. Upon resolution, it
        returns a string token from the first environment variable
        that is set.

        :param env_vars:
            A single environment variable or an ordered list of
            candidate environment variables.
        :param strict:
            Whether to raise an exception if none of the environment
            variables are set.
        """
        if isinstance(env_vars, str):
            env_vars = [env_vars]
        return EnvVarSecret(_env_vars=tuple(env_vars), _strict=strict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the secret to a JSON-serializable dictionary.
        Some secrets may not be serializable.

        :returns:
            The serialized policy.
        """
        out = {"type": self.type.value}
        inner = self._to_dict()
        assert all(k not in inner for k in out.keys())
        out.update(inner)
        return out

    @staticmethod
    def from_dict(dict: Dict[str, Any]) -> "Secret":
        """
        Create a secret from a JSON-serializable dictionary.

        :param dict:
            The dictionary with the serialized data.
        :returns:
            The deserialized secret.
        """
        secret_map = {SecretType.TOKEN: TokenSecret, SecretType.ENV_VAR: EnvVarSecret}
        secret_type = SecretType.from_str(dict["type"])
        return secret_map[secret_type]._from_dict(dict)  # type: ignore

    @abstractmethod
    def resolve_value(self) -> Optional[Any]:
        """
        Resolve the secret to an atomic value. The semantics
        of the value is secret-dependent.

        :returns:
            The value of the secret, if any.
        """
        pass

    @property
    @abstractmethod
    def type(self) -> SecretType:
        """
        The type of the secret.
        """
        pass

    @abstractmethod
    def _to_dict(self) -> Dict[str, Any]:
        pass

    @staticmethod
    @abstractmethod
    def _from_dict(dict: Dict[str, Any]) -> "Secret":
        pass


@dataclass(frozen=True)
class TokenSecret(Secret):
    """
    A secret that uses a string token/API key.
    Cannot be serialized.
    """

    _token: str
    _type: SecretType = SecretType.TOKEN

    def __post_init__(self):
        super().__init__()
        assert self._type == SecretType.TOKEN

        if len(self._token) == 0:
            raise ValueError("Authentication token cannot be empty.")

    def _to_dict(self) -> Dict[str, Any]:
        raise ValueError(
            "Cannot serialize token-based secret. Use an alternative secret type like environment variables."
        )

    @staticmethod
    def _from_dict(dict: Dict[str, Any]) -> "Secret":
        raise ValueError(
            "Cannot deserialize token-based secret. Use an alternative secret type like environment variables."
        )

    def resolve_value(self) -> Optional[Any]:
        return self._token

    @property
    def type(self) -> SecretType:
        return self._type


@dataclass(frozen=True)
class EnvVarSecret(Secret):
    """
    A secret that accepts one or more environment variables.
    Upon resolution, it returns a string token from the first
    environment variable that is set. Can be serialized.
    """

    _env_vars: Tuple[str, ...]
    _strict: bool = True
    _type: SecretType = SecretType.ENV_VAR

    def __post_init__(self):
        super().__init__()
        assert self._type == SecretType.ENV_VAR

        if len(self._env_vars) == 0:
            raise ValueError("One or more environment variables must be provided for the secret.")

    def _to_dict(self) -> Dict[str, Any]:
        return {"env_vars": list(self._env_vars), "strict": self._strict}

    @staticmethod
    def _from_dict(dict: Dict[str, Any]) -> "Secret":
        return EnvVarSecret(tuple(dict["env_vars"]), _strict=dict["strict"])

    def resolve_value(self) -> Optional[Any]:
        out = None
        for env_var in self._env_vars:
            value = os.getenv(env_var)
            if value is not None:
                out = value
                break
        if out is None and self._strict:
            raise ValueError(f"None of the following authentication environment variables are set: {self._env_vars}")
        return out

    @property
    def type(self) -> SecretType:
        return self._type


def deserialize_secrets_inplace(data: Dict[str, Any], keys: Iterable[str], *, recursive: bool = False):
    """
    Deserialize secrets in a dictionary inplace.

    :param data:
        The dictionary with the serialized data.
    :param keys:
        The keys of the secrets to deserialize.
    :param recursive:
        Whether to recursively deserialize nested dictionaries.
    """
    for k, v in data.items():
        if isinstance(v, dict) and recursive:
            deserialize_secrets_inplace(v, keys)
        elif k in keys and v is not None:
            data[k] = Secret.from_dict(v)


class SupportedAuthTypes(Enum):
    """
    Supported auth credentials for WeaviateDocumentStore.
    """

    API_KEY = "api_key"
    BEARER = "bearer"
    CLIENT_CREDENTIALS = "client_credentials"
    CLIENT_PASSWORD = "client_password"

    def __str__(self):
        return self.value

    @staticmethod
    def from_class(auth_class) -> "SupportedAuthTypes":
        auth_types = {
            AuthApiKey: SupportedAuthTypes.API_KEY,
            AuthBearerToken: SupportedAuthTypes.BEARER,
            AuthClientCredentials: SupportedAuthTypes.CLIENT_CREDENTIALS,
            AuthClientPassword: SupportedAuthTypes.CLIENT_PASSWORD,
        }
        return auth_types[auth_class]


@dataclass(frozen=True)
class AuthCredentials(ABC):
    """
    Base class for all auth credentials supported by WeaviateDocumentStore.
    Can be used to deserialize from dict any of the supported auth credentials.
    """

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the object to a dictionary representation for serialization.
        """
        _fields = {}
        for _field in fields(self):
            if _field.type is Secret:
                _fields[_field.name] = getattr(self, _field.name).to_dict()
            else:
                _fields[_field.name] = getattr(self, _field.name)

        return {"type": str(SupportedAuthTypes.from_class(self.__class__)), "init_parameters": _fields}

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "AuthCredentials":
        """
        Converts a dictionary representation to an auth credentials object.
        """
        if "type" not in data:
            msg = "Missing 'type' in serialization data"
            raise DeserializationError(msg)

        auth_classes: Dict[str, Type[AuthCredentials]] = {
            str(SupportedAuthTypes.API_KEY): AuthApiKey,
            str(SupportedAuthTypes.BEARER): AuthBearerToken,
            str(SupportedAuthTypes.CLIENT_CREDENTIALS): AuthClientCredentials,
            str(SupportedAuthTypes.CLIENT_PASSWORD): AuthClientPassword,
        }

        return auth_classes[data["type"]]._from_dict(data)

    @classmethod
    @abstractmethod
    def _from_dict(cls, data: Dict[str, Any]):
        """
        Internal method to convert a dictionary representation to an auth credentials object.
        All subclasses must implement this method.
        """

    @abstractmethod
    def resolve_value(self):
        """
        Resolves all the secrets in the auth credentials object and returns the corresponding Weaviate object.
        All subclasses must implement this method.
        """


@dataclass(frozen=True)
class AuthApiKey(AuthCredentials):
    """
    AuthCredentials for API key authentication.
    By default it will load `api_key` from the environment variable `WEAVIATE_API_KEY`.
    """

    api_key: Secret = field(default_factory=lambda: Secret.from_env_var(["WEAVIATE_API_KEY"]))

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "AuthApiKey":
        deserialize_secrets_inplace(data["init_parameters"], ["api_key"])
        return cls(**data["init_parameters"])

    def resolve_value(self) -> WeaviateAuthApiKey:
        return WeaviateAuthApiKey(api_key=self.api_key.resolve_value())


@dataclass(frozen=True)
class AuthBearerToken(AuthCredentials):
    """
    AuthCredentials for Bearer token authentication.
    By default it will load `access_token` from the environment variable `WEAVIATE_ACCESS_TOKEN`,
    and `refresh_token` from the environment variable
    `WEAVIATE_REFRESH_TOKEN`.
    `WEAVIATE_REFRESH_TOKEN` environment variable is optional.
    """

    access_token: Secret = field(default_factory=lambda: Secret.from_env_var(["WEAVIATE_ACCESS_TOKEN"]))
    expires_in: int = field(default=60)
    refresh_token: Secret = field(default_factory=lambda: Secret.from_env_var(["WEAVIATE_REFRESH_TOKEN"], strict=False))

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "AuthBearerToken":
        deserialize_secrets_inplace(data["init_parameters"], ["access_token", "refresh_token"])
        return cls(**data["init_parameters"])

    def resolve_value(self) -> WeaviateAuthBearerToken:
        access_token = self.access_token.resolve_value()
        refresh_token = self.refresh_token.resolve_value()

        return WeaviateAuthBearerToken(
            access_token=access_token,
            expires_in=self.expires_in,
            refresh_token=refresh_token,
        )


@dataclass(frozen=True)
class AuthClientCredentials(AuthCredentials):
    """
    AuthCredentials for client credentials authentication.
    By default it will load `client_secret` from the environment variable `WEAVIATE_CLIENT_SECRET`, and
    `scope` from the environment variable `WEAVIATE_SCOPE`.
    `WEAVIATE_SCOPE` environment variable is optional, if set it can either be a string or a list of space
    separated strings. e.g "scope1" or "scope1 scope2".
    """

    client_secret: Secret = field(default_factory=lambda: Secret.from_env_var(["WEAVIATE_CLIENT_SECRET"]))
    scope: Secret = field(default_factory=lambda: Secret.from_env_var(["WEAVIATE_SCOPE"], strict=False))

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "AuthClientCredentials":
        deserialize_secrets_inplace(data["init_parameters"], ["client_secret", "scope"])
        return cls(**data["init_parameters"])

    def resolve_value(self) -> WeaviateAuthClientCredentials:
        return WeaviateAuthClientCredentials(
            client_secret=self.client_secret.resolve_value(),
            scope=self.scope.resolve_value(),
        )


@dataclass(frozen=True)
class AuthClientPassword(AuthCredentials):
    """
    AuthCredentials for username and password authentication.
    By default it will load `username` from the environment variable `WEAVIATE_USERNAME`,
    `password` from the environment variable `WEAVIATE_PASSWORD`, and
    `scope` from the environment variable `WEAVIATE_SCOPE`.
    `WEAVIATE_SCOPE` environment variable is optional, if set it can either be a string or a list of space
    separated strings. e.g "scope1" or "scope1 scope2".
    """

    username: Secret = field(default_factory=lambda: Secret.from_env_var(["WEAVIATE_USERNAME"]))
    password: Secret = field(default_factory=lambda: Secret.from_env_var(["WEAVIATE_PASSWORD"]))
    scope: Secret = field(default_factory=lambda: Secret.from_env_var(["WEAVIATE_SCOPE"], strict=False))

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "AuthClientPassword":
        deserialize_secrets_inplace(data["init_parameters"], ["username", "password", "scope"])
        return cls(**data["init_parameters"])

    def resolve_value(self) -> WeaviateAuthClientPassword:
        return WeaviateAuthClientPassword(
            username=self.username.resolve_value(),
            password=self.password.resolve_value(),
            scope=self.scope.resolve_value(),
        )
