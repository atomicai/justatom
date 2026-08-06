import asyncio as asio
import base64
import datetime
import json
import os
from collections.abc import Generator, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import weaviate
from loguru import logger
from weaviate.classes.query import MetadataQuery
from weaviate.collections.classes.data import DataObject
from weaviate.config import AdditionalConfig
from weaviate.embedded import EmbeddedOptions
from weaviate.util import generate_uuid5

from justatom.etc.auth import AuthCredentials
from justatom.etc.errors import DocumentStoreError, DuplicateDocumentError
from justatom.etc.filters import convert_filters
from justatom.etc.schema import Document
from justatom.etc.serialization import default_from_dict, default_to_dict
from justatom.etc.types import DuplicatePolicy
from justatom.tooling.stl import AsyncConstructor

DOCUMENT_COLLECTION_PROPERTIES = [
    {"name": "_original_id", "dataType": ["text"]},
    {"name": "content", "dataType": ["text"]},
    {"name": "dataframe", "dataType": ["text"]},
    {"name": "blob_data", "dataType": ["blob"]},
    {"name": "blob_mime_type", "dataType": ["text"]},
    {"name": "score", "dataType": ["number"]},
    {
        "name": "meta",
        "dataType": ["object"],
        "nestedProperties": [
            {"dataType": ["text[]"], "name": "labels"},
            {"dataType": ["text"], "name": "url"},
            {
                "dataType": ["object[]"],
                "name": "keywords_or_phrases",
                "nestedProperties": [
                    {"name": "keyword_or_phrase", "dataType": ["text"]},
                    {"name": "explanation", "dataType": ["text"]},
                ],
            },
        ],
    },
]

DEFAULT_INVERTED_INDEX_CONFIG = {"bm25": {"b": 0.75, "k1": 1.2}}

DEFAULT_VECTOR_INDEX_CONFIG = {"vectorIndexConfig": {"distance": "dot"}}

DEFAULT_QUERY_LIMIT = 9999
DEFAULT_WEAVIATE_HOST = "localhost"
DEFAULT_WEAVIATE_PORT = 2211
DEFAULT_WEAVIATE_GRPC_PORT = 50051


def _to_documents_per_query(results: list[Any], converter) -> list[list[Document]]:
    response: list[list[Document]] = []
    for res in results:
        response.append([converter(doc) for doc in res.objects])
    return response


class WeaviateDocumentStore(AsyncConstructor):
    """
    `WeaviateDocumentStore` is a Document Store for Weaviate.
    It can be used with Weaviate Cloud Services or self-hosted instances.
    """

    async def __init__(
        self,
        url: str | None = None,
        collection_schema_name: str = "Default",
        auth_client_secret: Any | None = None,
        additional_headers: dict | None = None,
        embedded_options: EmbeddedOptions | None = None,
        additional_config: AdditionalConfig | None = None,
        grpc_port: int = 50051,
        grpc_secure: bool = False,
        **props,
    ):
        """
        Create a new instance of WeaviateDocumentStore and connects to the Weaviate instance.

        :param url:
            The URL to the weaviate instance.
        :param collection_settings:
            The collection settings to use. If `None`, it will use a collection named `default` with the following
            properties:
            - _original_id: text
            - content: text
            - dataframe: text
            - blob_data: blob
            - blob_mime_type: text
            - score: number
            The Document `meta` fields are omitted in the default collection settings as we can't make assumptions
            on the structure of the meta field.
            We heavily recommend to create a custom collection with the correct meta properties
            for your use case.
            Another option is relying on the automatic schema generation, but that's not recommended for
            production use.
            See the official `Weaviate documentation<https://weaviate.io/developers/weaviate/manage-data/collections>`_
            for more information on collections and their properties.
        :param auth_client_secret:
            Authentication credentials. Can be one of the following types depending on the authentication mode:
            - `AuthBearerToken` to use existing access and (optionally, but recommended) refresh tokens
            - `AuthClientPassword` to use username and password for oidc Resource Owner Password flow
            - `AuthClientCredentials` to use a client secret for oidc client credential flow
            - `AuthApiKey` to use an API key
        :param additional_headers:
            Additional headers to include in the requests. Can be used to set OpenAI/HuggingFace keys.
            OpenAI/HuggingFace key looks like this:
            ```
            {"X-OpenAI-Api-Key": "<THE-KEY>"}, {"X-HuggingFace-Api-Key": "<THE-KEY>"}
            ```
        :param embedded_options:
            If set, create an embedded Weaviate cluster inside the client. For a full list of options see
            `weaviate.embedded.EmbeddedOptions`.
        :param additional_config:
            Additional and advanced configuration options for weaviate.
        :param grpc_port:
            The port to use for the gRPC connection.
        :param grpc_secure:
            Whether to use a secure channel for the underlying gRPC API.
        """
        # proxies, timeout_config, trust_env are part of additional_config now
        # startup_period has been removed
        self._client = None
        self._close_task = None
        client_options = dict(props)
        client_options.setdefault("skip_init_checks", False)
        if auth_client_secret is not None:
            client_options["auth_client_secret"] = (
                auth_client_secret.resolve_value()
                if isinstance(auth_client_secret, AuthCredentials)
                else auth_client_secret
            )
        if additional_config is not None:
            client_options["additional_config"] = additional_config
        if additional_headers is not None:
            client_options["additional_headers"] = additional_headers
        if embedded_options is not None:
            client_options["embedded_options"] = embedded_options

        try:
            normalized_url = self._normalize_url(url) if url is not None else None
            connection_params = (
                weaviate.connect.base.ConnectionParams.from_url(
                    url=normalized_url,
                    grpc_port=grpc_port,
                    grpc_secure=grpc_secure,
                )
                if normalized_url is not None
                else None
            )
            self._client = weaviate.WeaviateAsyncClient(
                connection_params=connection_params,
                **client_options,
            )
            await self._client.connect()

            collection_schema_name = collection_schema_name.capitalize()
            self.collection_settings = {
                "class": collection_schema_name,
                "invertedIndexConfig": {"indexNullState": True},
                "properties": DOCUMENT_COLLECTION_PROPERTIES,
                "multiTenancyConfig": {"enabled": False},
            }

            collection_exists = await self._client.collections.exists(collection_schema_name)
            if not collection_exists:
                await self._client.collections.create_from_dict(self.collection_settings)
            self.__collection = self._client.collections.get(collection_schema_name)
            self.collection_name = self.__collection.name
        except BaseException:
            await self.close()
            raise

        self._url = normalized_url
        self._auth_client_secret = auth_client_secret
        self._additional_headers = additional_headers
        self._embedded_options = embedded_options
        self._additional_config = additional_config

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        embedded_options = asdict(self._embedded_options) if self._embedded_options else None
        additional_config = json.loads(self._additional_config.model_dump_json(by_alias=True)) if self._additional_config else None

        return default_to_dict(
            self,
            url=self._url,
            collection_settings=self.collection_settings,
            auth_client_secret=(self._auth_client_secret.to_dict() if self._auth_client_secret else None),
            additional_headers=self._additional_headers,
            embedded_options=embedded_options,
            additional_config=additional_config,
        )

    @staticmethod
    def _normalize_host(value: Any, default: str = DEFAULT_WEAVIATE_HOST) -> str:
        if value is None:
            return default
        host = str(value).strip()
        if host == "":
            return default
        if host.lower() in {"none", "null"}:
            return default
        if host.startswith("${") and host.endswith("}"):
            return default
        return host

    @staticmethod
    def _normalize_port(
        value: Any,
        *,
        default: int,
        setting_name: str,
    ) -> int:
        if value is None:
            return default
        if isinstance(value, str):
            stripped = value.strip()
            if stripped == "":
                return default
            if stripped.lower() in {"none", "null"}:
                return default
            if stripped.startswith("${") and stripped.endswith("}"):
                return default
            value = stripped

        try:
            port = int(value)
        except (TypeError, ValueError) as exc:
            raise DocumentStoreError(f"Invalid {setting_name}={value!r}. Expected a positive integer port.") from exc

        if port <= 0:
            raise DocumentStoreError(f"Invalid {setting_name}={value!r}. Expected a positive integer port.")

        return port

    @staticmethod
    def _normalize_url(url: str) -> str:
        try:
            parsed = urlsplit(url)
            hostname = parsed.hostname
            port = parsed.port
        except ValueError as exc:
            raise DocumentStoreError("Invalid Weaviate URL: malformed host or port") from exc

        if parsed.scheme.lower() not in {"http", "https"}:
            raise DocumentStoreError("Invalid Weaviate URL: scheme must be http or https")
        if hostname is None:
            raise DocumentStoreError("Invalid Weaviate URL: host is required")
        if parsed.username is not None or parsed.password is not None:
            raise DocumentStoreError("Invalid Weaviate URL: userinfo is not allowed")
        if parsed.path not in {"", "/"}:
            raise DocumentStoreError("Invalid Weaviate URL: paths are not allowed")
        if parsed.query:
            raise DocumentStoreError("Invalid Weaviate URL: query parameters are not allowed")
        if parsed.fragment:
            raise DocumentStoreError("Invalid Weaviate URL: fragments are not allowed")

        normalized_host = f"[{hostname}]" if ":" in hostname else hostname
        netloc = normalized_host if port is None else f"{normalized_host}:{port}"
        return urlunsplit((parsed.scheme.lower(), netloc, "", "", ""))

    @classmethod
    async def connect(
        cls,
        collection: str,
        url: str | None = None,
        grpc_port: int = DEFAULT_WEAVIATE_GRPC_PORT,
        grpc_secure: bool = False,
        **client_options: Any,
    ) -> "WeaviateDocumentStore":
        if "connection_params" in client_options:
            raise DocumentStoreError("connection_params conflicts with the url and gRPC connection options")

        if url is None:
            weaviate_host = cls._normalize_host(os.environ.get("WEAVIATE_HOST"), default=DEFAULT_WEAVIATE_HOST)
            weaviate_port = cls._normalize_port(
                os.environ.get("WEAVIATE_PORT"),
                default=DEFAULT_WEAVIATE_PORT,
                setting_name="WEAVIATE_PORT",
            )
            url = f"http://{weaviate_host}:{weaviate_port}"

        url = cls._normalize_url(url)

        normalized_grpc_port = cls._normalize_port(
            grpc_port,
            default=DEFAULT_WEAVIATE_GRPC_PORT,
            setting_name="grpc_port",
        )
        logger.info("WEAVIATE | connecting collection=[{}]", collection)
        try:
            store = await cls(
                collection_schema_name=collection,
                url=url,
                grpc_port=normalized_grpc_port,
                grpc_secure=grpc_secure,
                **client_options,
            )  # type: ignore
        except asio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "WEAVIATE | connection failed for url=[{}] grpc_port=[{}] collection=[{}]",
                url,
                normalized_grpc_port,
                collection,
            )
            raise DocumentStoreError(
                "Failed to connect to Weaviate at "
                f"{url} (grpc_port={normalized_grpc_port}) for collection "
                f"{collection!r}. Check `WEAVIATE_HOST`, `WEAVIATE_PORT`, "
                "the explicit URL, and whether Weaviate is running."
            ) from exc
        return store

    async def _ensure_async_connection(self) -> None:
        if self._client is None:
            raise DocumentStoreError("Async Weaviate client is not initialised")
        if not self._client.is_connected():
            try:
                await self._client.connect()
            except Exception as exc:
                raise DocumentStoreError("Failed to reconnect async Weaviate client") from exc

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WeaviateDocumentStore":
        """
        Deserializes the component from a dictionary.

        :param data:
            The dictionary to deserialize from.
        :returns:
            The deserialized component.
        """
        if (auth_client_secret := data["init_parameters"].get("auth_client_secret")) is not None:
            data["init_parameters"]["auth_client_secret"] = AuthCredentials.from_dict(auth_client_secret)
        if (embedded_options := data["init_parameters"].get("embedded_options")) is not None:
            data["init_parameters"]["embedded_options"] = EmbeddedOptions(**embedded_options)
        if (additional_config := data["init_parameters"].get("additional_config")) is not None:
            data["init_parameters"]["additional_config"] = AdditionalConfig(**additional_config)
        return default_from_dict(
            cls,
            data,
        )

    async def delete_collection(self):
        await self._ensure_async_connection()
        await self._client.collections.delete(self.__collection.name)

    async def count_documents(self) -> int:
        """
        Returns the number of documents present in the DocumentStore.
        """
        await self._ensure_async_connection()
        total = await self.__collection.aggregate.over_all(total_count=True)
        return total.total_count if total else 0  # type: ignore

    def _to_data_object(self, document: Document) -> dict[str, Any]:
        """
        Converts a Document to a Weaviate data object ready to be saved.
        """
        data = deepcopy(document.to_dict())
        # Weaviate forces a UUID as an id.
        # We don't know if the id of our Document is a UUID or not, so we save it on a different field
        # and let Weaviate a UUID that we're going to ignore completely.
        data["_original_id"] = data.pop("id")
        blob = data.pop("blob", None)
        if blob is not None:
            # Weaviate wants the blob data as a base64 encoded string
            # See the official docs for more information:
            # https://weaviate.io/developers/weaviate/config-refs/datatypes#datatype-blob
            data["blob_data"] = base64.b64encode(bytes(blob.pop("data"))).decode()
            data["blob_mime_type"] = blob.pop("mime_type")
        # The embedding vector is stored separately from the rest of the data
        del data["embedding"]

        if "sparse_embedding" in data:
            sparse_embedding = data.pop("sparse_embedding", None)
            if sparse_embedding:
                logger.warning(
                    "Document %s has the `sparse_embedding` field set,"
                    "but storing sparse embeddings in Weaviate is not currently supported."
                    "The `sparse_embedding` field will be ignored.",
                    data["_original_id"],
                )

        AVAILABLE_PROPS = set([f["name"] for f in DOCUMENT_COLLECTION_PROPERTIES])
        # Delete all the rest keys
        if "meta" in data.keys() and "meta" not in AVAILABLE_PROPS:  # noqa: SIM118
            logger.warning(f"[meta={data['meta']}] is present and will be ignored since it is NOT registred in a collection.")
            del data["meta"]

        return data

    def _to_document(self, data: DataObject[dict[str, Any], None]) -> Document:
        """
        Converts a data object read from Weaviate into a Document.
        """
        document_data = deepcopy(data.properties)
        document_data["id"] = document_data.pop("_original_id")
        if isinstance(data.vector, list):
            document_data["embedding"] = deepcopy(data.vector)
        elif isinstance(data.vector, dict):
            document_data["embedding"] = deepcopy(data.vector.get("default"))
        else:
            document_data["embedding"] = None

        if (blob_data := document_data.get("blob_data")) is not None:
            document_data["blob"] = {
                "data": base64.b64decode(blob_data),
                "mime_type": document_data.get("blob_mime_type"),
            }

        # We always delete these fields as they're not part of the Document dataclass
        document_data.pop("blob_data", None)
        document_data.pop("blob_mime_type", None)

        for key, value in document_data.items():
            if isinstance(value, datetime.datetime):
                document_data[key] = value.strftime("%Y-%m-%dT%H:%M:%SZ")
        weaviate_meta = getattr(data, "metadata", None)
        score = getattr(weaviate_meta, "score", None)
        certainty = getattr(weaviate_meta, "certainty", None)
        if score is not None:
            document_data["score"] = score
        elif certainty is not None:
            document_data["score"] = certainty

        return Document.from_dict(document_data)

    def _check_keywords(self, docs: list[Document], keywords: list[str] | None = None):
        # TODO: Rewrite using custom handler for every single item. e.g. class Response
        response = docs
        if keywords:
            response = [doc for doc in response if any([kw in doc.keywords for kw in keywords])]  # type: ignore
        return response

    async def _query(self) -> list[dict[str, Any]]:
        # properties = [p.name for p in self._collection.config.get().properties]
        result = []
        try:
            async for obj in self.__collection.iterator(include_vector=True, return_properties=None):
                result.append(obj)
        except weaviate.exceptions.WeaviateQueryError as e:
            msg = f"Failed to query documents in Weaviate. Error: {e.message}"
            raise DocumentStoreError(msg) from e
        return result

    async def _query_with_filters(self, filters: dict[str, Any]) -> list[dict[str, Any]]:
        # properties = [p.name for p in self._collection.config.get().properties]
        # When querying with filters we need to paginate using limit and offset as using
        # a cursor with after is not possible. See the official docs:
        # https://weaviate.io/developers/weaviate/api/graphql/additional-operators#cursor-with-after
        #
        # Nonetheless there's also another issue, paginating with limit and offset is not efficient
        # and it's still restricted by the QUERY_MAXIMUM_RESULTS environment variable.
        # If the sum of limit and offest is greater than QUERY_MAXIMUM_RESULTS an error is raised.
        # See the official docs for more:
        # https://weaviate.io/developers/weaviate/api/graphql/additional-operators#performance-considerations
        await self._ensure_async_connection()
        offset = 0
        partial_result = None
        result = []
        # Keep querying until we get all documents matching the filters
        while partial_result is None or len(partial_result.objects) == DEFAULT_QUERY_LIMIT:
            try:
                partial_result = await self.__collection.query.fetch_objects(
                    filters=convert_filters(filters),
                    include_vector=True,
                    limit=DEFAULT_QUERY_LIMIT,
                    offset=offset,
                    return_properties=None,
                )
            except weaviate.exceptions.WeaviateQueryError as e:
                msg = f"Failed to query documents in Weaviate. Error: {e.message}"
                raise DocumentStoreError(msg) from e
            result.extend(partial_result.objects)
            offset += DEFAULT_QUERY_LIMIT
        return result

    async def filter_documents(self, filters: dict[str, Any] | None = None) -> list[Document]:
        """
        Returns the documents that match the filters provided.

        For a detailed specification of the filters, refer to the
        DocumentStore.filter_documents() protocol documentation.

        :param filters: The filters to apply to the document list.
        :returns: A list of Documents that match the given filters.
        """
        result = []
        if filters:  # noqa: SIM108
            result = await self._query_with_filters(filters)
        else:
            result = await self._query()
        return [self._to_document(doc) for doc in result]  # type: ignore

    async def _batch_write(self, documents: list[Document], policy: DuplicatePolicy, batch_size: int = 64) -> int:
        """
        Writes document to Weaviate in batches.
        Documents with the same id will be overwritten.
        Raises in case of errors.
        """
        await self._ensure_async_connection()
        wrapped_documents = [
            DataObject(
                properties=self._to_data_object(doc),
                uuid=generate_uuid5(doc.id),
                vector=doc.embedding,
            )
            for doc in documents
        ]
        try:
            batch_response = await self.__collection.data.insert_many(wrapped_documents)
        except weaviate.exceptions.UnexpectedStatusCodeError as error:
            msg = f"Error writing documents to Weaviate: {str(error)}"
            raise DocumentStoreError(msg) from error
        else:
            n_written_docs = len(wrapped_documents) - len(batch_response.errors)
        return n_written_docs

    def get_collection_name(self):
        return self.__collection.name

    async def _write(self, documents: list[Document], policy: DuplicatePolicy) -> int:
        """
        Writes documents to Weaviate using the specified policy.
        This doesn't uses the batch API, so it's slower than _batch_write.
        If policy is set to SKIP it will skip any document that already exists.
        If policy is set to FAIL it will raise an exception if any of the documents already exists.
        """
        written = 0
        duplicate_errors_ids = []
        for doc in documents:
            if not isinstance(doc, Document):
                msg = f"Expected a Document, got '{type(doc)}' instead."
                raise ValueError(msg)

            does_exist = await self.__collection.data.exists(uuid=generate_uuid5(doc.id))
            if policy == DuplicatePolicy.SKIP and does_exist:
                # This Document already exists, we skip it
                continue

            try:
                await self.__collection.data.insert(
                    uuid=generate_uuid5(doc.id),
                    properties=self._to_data_object(doc),
                    vector=doc.embedding,  # type: ignore
                )

                written += 1
            except weaviate.exceptions.UnexpectedStatusCodeError:
                if policy == DuplicatePolicy.FAIL:
                    duplicate_errors_ids.append(str(doc.id))
        if duplicate_errors_ids:
            msg = f"IDs '{', '.join(duplicate_errors_ids)}' already exist in the document store."
            raise DuplicateDocumentError(msg)
        return written

    async def write_documents(self, documents: Sequence[Document]) -> int:
        """
        Writes documents to Weaviate using the specified policy.
        We recommend using a OVERWRITE policy as it's faster than other policies for Weaviate since it uses
        the batch API.
        We can't use the batch API for other policies as it doesn't return any information whether the document
        already exists or not. That prevents us from returning errors when using the FAIL policy or skipping a
        Document when using the SKIP policy.
        """
        await self._ensure_async_connection()
        total_written_docs = await self._batch_write(
            list(documents),
            batch_size=128,
            policy=DuplicatePolicy.OVERWRITE,
        )
        return total_written_docs

    async def get_all_documents(self, include_vector: bool = False) -> Generator:  # type: ignore
        await self._ensure_async_connection()
        props = dict(include_vector=include_vector)
        async for obj in self.__collection.iterator(**props):  # type: ignore # noqa: UP028
            yield obj  # type: ignore

    def get_document_by_id(self):
        pass

    async def get_all_documents_by_ids(
        self, document_ids: str | list[str], include_vector: bool = False
    ) -> Generator:  # type: ignore
        await self._ensure_async_connection()
        document_ids = [document_ids] if isinstance(document_ids, str) else document_ids
        for document_id in document_ids:
            js_document_id = generate_uuid5(document_id)
            js_single_response = await self.__collection.query.fetch_object_by_id(js_document_id, include_vector=include_vector)
            if js_single_response is not None:
                yield self._to_document(js_single_response)  # type: ignore

    async def delete_documents(self, document_ids: list[str]) -> None:
        """
        Deletes all documents with matching document_ids from the DocumentStore.

        :param document_ids: The object_ids to delete.
        """
        await self._ensure_async_connection()
        weaviate_ids = [generate_uuid5(doc_id) for doc_id in document_ids]
        result = await self.__collection.data.delete_many(
            where=weaviate.classes.query.Filter.by_id().contains_any(weaviate_ids)
        )
        if result.failed > 0:
            raise DocumentStoreError(
                f"Weaviate failed to delete {result.failed} of {result.matches} matched documents"
            )

    async def clear(self) -> None:
        collection_name = self.collection_settings.get("class")
        try:
            await self._ensure_async_connection()
            ids = []
            async for obj in self.get_all_documents():
                maybe_original_id = obj.properties.get("_original_id")
                if maybe_original_id is not None:
                    ids.append(maybe_original_id)

            if len(ids) == 0:
                logger.info(f"Nothing to delete in {self.__collection.name}")
                return

            await self.delete_documents(document_ids=ids)
        except asio.CancelledError:
            raise
        except Exception as exc:
            raise DocumentStoreError(f"Error deleting documents for {collection_name}") from exc

    async def search_keywords(
        self,
        queries: Sequence[str],
        *,
        top_k: int,
        filters: Mapping[str, Any] | None = None,
    ) -> list[list[Document]]:
        logger.info(f"SEARCH | algo=[BM25] | collection_name=[{self.__collection.name}]")
        queries = list(queries)
        if not queries:
            return []
        await self._ensure_async_connection()
        weaviate_filters = convert_filters(dict(filters)) if filters else None
        result = await asio.gather(
            *(
                self.__collection.query.bm25(
                    query=query,
                    filters=weaviate_filters,
                    limit=top_k,
                    include_vector=False,
                    query_properties=["content"],
                    return_properties=None,
                    return_metadata=MetadataQuery(distance=True, score=True, explain_score=True, certainty=True),
                )
                for query in queries
            )
        )
        return _to_documents_per_query(result, self._to_document)

    async def search_hybrid(
        self,
        queries: Sequence[str],
        vectors: Sequence[Sequence[float]],
        *,
        alpha: float,
        top_k: int,
        filters: Mapping[str, Any] | None = None,
        include_vectors: bool = False,
    ) -> list[list[Document]]:
        queries = list(queries)
        vectors = list(vectors)
        if len(queries) != len(vectors):
            raise ValueError(f"Expected {len(queries)} vectors, received {len(vectors)}")
        if not queries:
            return []
        await self._ensure_async_connection()
        weaviate_filters = convert_filters(dict(filters)) if filters else None
        result = await asio.gather(
            *(
                self.__collection.query.hybrid(
                    query=query,
                    vector=vector,
                    alpha=alpha,
                    limit=top_k,
                    filters=weaviate_filters,
                    return_metadata=MetadataQuery(distance=True, score=True, explain_score=True, certainty=True),
                    include_vector=include_vectors,
                    query_properties=["content"],
                )
                for query, vector in zip(queries, vectors, strict=True)
            )
        )
        return _to_documents_per_query(result, self._to_document)

    async def search_vector(
        self,
        vectors: Sequence[Sequence[float]],
        *,
        top_k: int,
        filters: Mapping[str, Any] | None = None,
        include_vectors: bool = False,
    ) -> list[list[Document]]:
        vectors = list(vectors)
        if not vectors:
            return []
        await self._ensure_async_connection()
        weaviate_filters = convert_filters(dict(filters)) if filters else None
        result = await asio.gather(
            *(
                self.__collection.query.near_vector(
                    near_vector=vector,
                    include_vector=include_vectors,
                    filters=weaviate_filters,
                    limit=top_k,
                    return_properties=None,
                    return_metadata=["certainty"],
                )
                for vector in vectors
            )
        )
        return _to_documents_per_query(result, self._to_document)

    async def close(self) -> None:
        close_task = getattr(self, "_close_task", None)
        if close_task is None:
            client = getattr(self, "_client", None)
            if client is None:
                return
            close_task = asio.create_task(client.close())
            self._close_task = close_task

        try:
            await asio.shield(close_task)
        except asio.CancelledError as cancellation:
            try:
                await asio.shield(close_task)
            except BaseException as cleanup_error:
                if self._close_task is close_task:
                    self._close_task = None
                raise cleanup_error from cancellation
            if self._close_task is close_task:
                self._client = None
                self._close_task = None
            raise
        except BaseException:
            if self._close_task is close_task:
                self._close_task = None
            raise
        else:
            if self._close_task is close_task:
                self._client = None
                self._close_task = None


__all__ = ["WeaviateDocumentStore"]
