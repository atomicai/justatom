import asyncio as asio
import base64
import copy
import datetime
import json
import os
from collections.abc import Generator
from dataclasses import asdict
from typing import Any

import weaviate
from loguru import logger
from more_itertools import chunked
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
from justatom.etc.types import DuplicatePolicy, SearchPolicy
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
            {"dataType": ["text[]"], "name": "queries"},
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


class WeaviateDocStore(AsyncConstructor):
    """
    `WeaviateDocumentStore` is a Document Store for Weaviate.
    It can be used with Weaviate Cloud Services or self-hosted instances.
    """

    async def __init__(
        self,
        url: str | None = None,
        collection_schema_name: str = "Default",
        auth_client_secret: AuthCredentials | None = None,
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
        self._client = weaviate.WeaviateAsyncClient(
            connection_params=(
                weaviate.connect.base.ConnectionParams.from_url(
                    url=url, grpc_port=grpc_port, grpc_secure=grpc_secure
                )
                if url
                else None
            ),
            auth_client_secret=(
                auth_client_secret.resolve_value() if auth_client_secret else None
            ),
            additional_config=additional_config,
            additional_headers=additional_headers,
            embedded_options=embedded_options,
            skip_init_checks=False,
        )
        await self._client.connect()

        self._sync_client = weaviate.WeaviateClient(
            connection_params=(
                weaviate.connect.base.ConnectionParams.from_url(
                    url=url, grpc_port=grpc_port, grpc_secure=grpc_secure
                )
                if url
                else None
            ),
            auth_client_secret=(
                auth_client_secret.resolve_value() if auth_client_secret else None
            ),
            additional_config=additional_config,
            additional_headers=additional_headers,
            embedded_options=embedded_options,
            skip_init_checks=False,
        )

        self._sync_client.connect()

        # Test connection, it will raise an exception if it fails.
        # TODO: Re=parametrize to make it friendly for hybrid-search via bm25 + embedding search.
        collection_schema_name = collection_schema_name.capitalize()
        self.collection_settings = {
            "class": f"{collection_schema_name}",
            "invertedIndexConfig": {"indexNullState": True},
            "properties": DOCUMENT_COLLECTION_PROPERTIES,
            "multiTenancyConfig": {"enabled": False},
        }

        collection_exist = await self._client.collections.exists(collection_schema_name)
        if not collection_exist:
            await self._client.collections.create_from_dict(self.collection_settings)
        self._url = url
        self._auth_client_secret = auth_client_secret
        self._additional_headers = additional_headers
        self._embedded_options = embedded_options
        self._additional_config = additional_config
        self.__collection = self._client.collections.get(collection_schema_name)
        self.collection_name = self.__collection.name

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        embedded_options = (
            asdict(self._embedded_options) if self._embedded_options else None
        )
        additional_config = (
            json.loads(self._additional_config.model_dump_json(by_alias=True))
            if self._additional_config
            else None
        )

        return default_to_dict(
            self,
            url=self._url,
            collection_settings=self.collection_settings,
            auth_client_secret=(
                self._auth_client_secret.to_dict() if self._auth_client_secret else None
            ),
            additional_headers=self._additional_headers,
            embedded_options=embedded_options,
            additional_config=additional_config,
        )

    @classmethod
    async def connect(cls, collection_schema_name: str, **kwargs):
        WEAVIATE_HOST = kwargs.get("WEAVIATE_HOST") or os.environ.get("WEAVIATE_HOST")
        WEAVIATE_PORT = kwargs.get("WEAVIATE_PORT") or os.environ.get("WEAVIATE_PORT")
        WEAVIATE_GRPC_PORT = kwargs.get("WEAVIATE_GRPC_PORT") or os.environ.get(
            "WEAVIATE_GRPC_PORT"
        ) or "50051"
        logger.info(f"FINDER | collection_schema_name=[{collection_schema_name}]")
        store = await cls(
            collection_schema_name=collection_schema_name,
            url=f"http://{WEAVIATE_HOST}:{WEAVIATE_PORT}",
            grpc_port=int(WEAVIATE_GRPC_PORT),  # type: ignore
        )
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
    def from_dict(cls, data: dict[str, Any]) -> "WeaviateDocStore":
        """
        Deserializes the component from a dictionary.

        :param data:
            The dictionary to deserialize from.
        :returns:
            The deserialized component.
        """
        if (
            auth_client_secret := data["init_parameters"].get("auth_client_secret")
        ) is not None:
            data["init_parameters"]["auth_client_secret"] = AuthCredentials.from_dict(
                auth_client_secret
            )
        if (
            embedded_options := data["init_parameters"].get("embedded_options")
        ) is not None:
            data["init_parameters"]["embedded_options"] = EmbeddedOptions(
                **embedded_options
            )
        if (
            additional_config := data["init_parameters"].get("additional_config")
        ) is not None:
            data["init_parameters"]["additional_config"] = AdditionalConfig(
                **additional_config
            )
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
        data = document.to_dict()
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
            logger.warning(
                f"[meta={data['meta']}] is present and will be ignored since it is NOT registred in a collection."
            )
            del data["meta"]

        return data

    def _to_document(self, data: DataObject[dict[str, Any], None]) -> Document:
        """
        Converts a data object read from Weaviate into a Document.
        """
        document_data = data.properties
        document_data["id"] = document_data.pop("_original_id")
        if isinstance(data.vector, list):
            document_data["embedding"] = data.vector
        elif isinstance(data.vector, dict):
            document_data["embedding"] = data.vector.get("default")
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
        if getattr(weaviate_meta, "score", None) is not None:
            # Depending on the type of retrieval we get score from different fields.
            # score is returned when using BM25 retrieval.
            # certainty is returned when using embedding retrieval.
            # TODO: When using hybrid search
            if weaviate_meta.score is not None:  # type: ignore
                document_data["score"] = weaviate_meta.score  # type: ignore
            elif weaviate_meta.certainty is not None:  # type: ignore
                document_data["score"] = weaviate_meta.certainty  # type: ignore

        return Document.from_dict(document_data)

    def _check_keywords(self, docs: list[Document], keywords: list[str] | None = None):
        # TODO: Rewrite using custom handler for every single item. e.g. class Response
        response = docs
        if keywords:
            response = [
                doc for doc in response if any([kw in doc.keywords for kw in keywords])  # type: ignore
            ]
        return response

    async def _query(self) -> list[dict[str, Any]]:
        # properties = [p.name for p in self._collection.config.get().properties]
        try:
            result = await self.__collection.iterator(
                include_vector=True, return_properties=None
            )  # type: ignore
        except weaviate.exceptions.WeaviateQueryError as e:
            msg = f"Failed to query documents in Weaviate. Error: {e.message}"
            raise DocumentStoreError(msg) from e
        return result

    async def _query_with_filters(
        self, filters: dict[str, Any]
    ) -> list[dict[str, Any]]:
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
        while (
            partial_result is None
            or len(partial_result.objects) == DEFAULT_QUERY_LIMIT
        ):
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

    async def filter_documents(
        self, filters: dict[str, Any] | None = None
    ) -> list[Document]:
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
        

    async def _batch_write(
        self, documents: list[Document], policy: DuplicatePolicy, batch_size: int = 64
    ) -> int:
        """
        Writes document to Weaviate in batches.
        Documents with the same id will be overwritten.
        Raises in case of errors.
        """
        await self._ensure_async_connection()
        wrapped_documents = [DataObject(properties=self._to_data_object(doc), uuid=generate_uuid5(doc.id), vector=doc.embedding) for doc in documents]
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

            does_exist = await self.__collection.data.exists(
                uuid=generate_uuid5(doc.id)
            )
            if policy == DuplicatePolicy.SKIP and does_exist:
                # This Document already exists, we skip it
                continue

            try:
                reference_uuid_object = await self.__collection.data.insert(
                    uuid=generate_uuid5(doc.id),
                    properties=self._to_data_object(doc),
                    vector=doc.embedding,  # type: ignore
                )

                written += 1
            except weaviate.exceptions.UnexpectedStatusCodeError:
                if policy == DuplicatePolicy.FAIL:
                    duplicate_errors_ids.append(reference_uuid_object)
        if duplicate_errors_ids:
            msg = f"IDs '{', '.join(duplicate_errors_ids)}' already exist in the document store."
            raise DuplicateDocumentError(msg)
        return written

    async def write_documents(
        self,
        documents: list[Document],
        policy: DuplicatePolicy = DuplicatePolicy.OVERWRITE,
        batch_size: int = 128,
    ) -> int:
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
            documents, batch_size=batch_size, policy=policy
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
        document_ids = (
            [document_ids] if isinstance(document_ids, str) else document_ids
        )
        for document_id in document_ids:
            js_document_id = generate_uuid5(document_id)
            js_single_response = await self.__collection.query.fetch_object_by_id(
                js_document_id, include_vector=include_vector
            )
            if js_single_response is not None:
                yield self._to_document(js_single_response)  # type: ignore

    async def delete_documents(self, document_ids: list[str]) -> None:
        """
        Deletes all documents with matching document_ids from the DocumentStore.

        :param document_ids: The object_ids to delete.
        """
        await self._ensure_async_connection()
        weaviate_ids = [generate_uuid5(doc_id) for doc_id in document_ids]
        await self.__collection.data.delete_many(
            where=weaviate.classes.query.Filter.by_id().contains_any(weaviate_ids)
        )

    async def delete_all_documents(self) -> bool:
        await self._ensure_async_connection()
        ids = []
        async for x in self.get_all_documents():
            ids += [x.properties["_original_id"]]
            if len(ids) > 0:
                try:
                    await self.delete_documents(document_ids=ids)
                except:  # noqa: E722
                    logger.error(
                        f"Error deleting documents for {self.collection_settings.get('class')}, see logs for more details."
                    )
                    return False
                else:
                    return True
            logger.info(f"Nothing to delete in {self.__collection.name}")
            return True

    def _ensure_sync_connection(self) -> None:
        if self._sync_client is None:
            raise DocumentStoreError("Sync Weaviate client is not initialised")
        if not self._sync_client.is_connected():
            try:
                self._sync_client.connect()
            except Exception as exc:
                raise DocumentStoreError("Failed to reconnect sync Weaviate client") from exc
    
    def search_by_keywords_sync(
        self,
        queries: str | list[str],
        policy: SearchPolicy | None = SearchPolicy.BM25,
        filters: dict[str, Any] | None = None,
        keywords: list | None = None,
        top_k: int | None = None,
        include_vector: bool | None = False,
    ) -> list[Document]:
        queries = [queries] if isinstance(queries, str) else queries
        self._ensure_sync_connection()
        collection = self._sync_client.collections.get(self.__collection.name)
        response = []
        for q in queries:
            if policy == SearchPolicy.BM25:
                result = collection.query.bm25(
                        query=q,
                        filters=convert_filters(filters) if filters else None,
                        limit=top_k,
                        query_properties=["content"],
                        return_properties=None,
                        return_metadata=MetadataQuery(
                            distance=True, score=True, explain_score=True, certainty=True
                        ),
                    )  # type: ignore
                response.append([self._to_document(doc) for doc in result.objects])
            else:
                msg = f"You specified {str(policy)} that is not compatable with [search_by_keywords]. Only [BM25] is avalaible"
                logger.error(msg)
                raise ValueError(msg)
        return response
    

    async def search_by_keywords(
        self,
        queries: str | list[str],
        policy: SearchPolicy | None = SearchPolicy.BM25,
        filters: dict[str, Any] | None = None,
        keywords: list | None = None,
        top_k: int | None = None,
        include_vector: bool | None = False,
    ) -> list[Document]:
        # properties = [p.name for p in self._collection.config.get().properties]
        logger.info(
            f"SEARCH | algo=[BM25] | collection_name=[{self.__collection.name}]"
        )
        queries = [queries] if isinstance(queries, str) else queries
        await self._ensure_async_connection()
        if policy == SearchPolicy.BM25:
            if len(queries) <= 1:
                result = await self.__collection.query.bm25(
                    query=queries[0],
                    filters=convert_filters(filters) if filters else None,
                    limit=top_k,
                    include_vector=include_vector,  # type: ignore
                    query_properties=["content"],
                    return_properties=None,
                    return_metadata=MetadataQuery(
                        distance=True, score=True, explain_score=True, certainty=True
                    ),
                )  # type: ignore
                result = [result]
            else:
                parallel_workers = [
                    self.__collection.query.bm25(  # type: ignore
                        query=query,
                        filters=convert_filters(filters) if filters else None,
                        limit=top_k,
                        include_vector=include_vector,  # type: ignore
                        query_properties=["content"],
                        return_properties=None,
                        return_metadata=MetadataQuery(
                            distance=True, score=True, explain_score=True, certainty=True
                        ),
                    )
                    for query in queries
                ]
                result = await asio.gather(*parallel_workers)
        else:
            msg = f"You specified {str(policy)} that is not compatable with [search_by_keywords]. Only [BM25] is avalaible"
            logger.error(msg)
            raise ValueError(msg)
        response = [self._to_document(doc) for res in result for doc in res.objects]
        return response

    async def search(
        self,
        queries: str | list[str],
        queries_embeddings: list[float] | list[list[float]],
        rank_policy: str | None = None,
        alpha: float | None = 0.22,
        filters: dict[str, Any] | None = None,
        keywords: list | None = None,
        top_k: int | None = None,
        return_metadata: list[str] | None = None,
        include_vector: bool | None = False,
    ) -> list[Document]:
        """
        This method assumes the hybrid search with one of the present `ranking` methods out there.
        """
        return_metadata = (
            MetadataQuery(distance=True, score=True, explain_score=True, certainty=True)
            if return_metadata is None
            else return_metadata
        )  # type: ignore
        queries = [queries] if isinstance(queries, str) else queries
        queries_embeddings = (
            [queries_embeddings]
            if isinstance(queries_embeddings[0], float)
            else queries_embeddings
        )
        assert len(queries) == len(queries_embeddings), (
            f"Mismatch in number of queries and embeddings provided. "
            f"queries=[{len(queries)}] | embeddings=[{len(queries_embeddings)}]"
        )
        await self._ensure_async_connection()

        if len(queries) <= 1:
            result = await self.__collection.query.hybrid(
                query=queries[0],
                vector=queries_embeddings[0],
                alpha=alpha,  # type: ignore
                limit=top_k,
                filters=convert_filters(filters) if filters else None,
                return_metadata=return_metadata,  # type: ignore
                include_vector=include_vector,  # type: ignore
                query_properties=["content"],
            )  # type: ignore
            result = [result]
        else:
            parallel_workers = [
                self.__collection.query.hybrid(  # type: ignore
                    query=query,
                    vector=query_embedding,
                    alpha=alpha,  # type: ignore
                    limit=top_k,
                    filters=convert_filters(filters) if filters else None,
                    return_metadata=return_metadata,  # type: ignore
                    include_vector=include_vector,  # type: ignore
                    query_properties=["content"],
                )
                for query, query_embedding in zip(
                    queries, queries_embeddings, strict=False
                )
            ]
            result = await asio.gather(*parallel_workers)
        response = [self._to_document(doc) for res in result for doc in res.objects]
        return response

    async def search_by_embedding(
        self,
        queries_embeddings: list[float] | list[list[float]],
        filters: dict[str, Any] | None = None,
        keywords: list[str] | None = None,
        top_k: int | None = None,
        distance: float | None = None,
        certainty: float | None = None,
        return_metadata: list[str] | None = None,
        include_vector: bool | None = False,
    ) -> list[Document]:
        if distance is not None and certainty is not None:
            msg = "Can't use 'distance' and 'certainty' parameters together"
            raise ValueError(msg)
        queries_embeddings = [queries_embeddings] if isinstance(queries_embeddings[0], float) else queries_embeddings
        return_metadata = ["certainty"] if return_metadata is None else return_metadata
        # properties = [p.name for p in self._collection.config.get().properties]
        await self._ensure_async_connection()
        if len(queries_embeddings) <= 1:
            result = await self.__collection.query.near_vector(
                near_vector=queries_embeddings[0],
                distance=distance,
                certainty=certainty,
                include_vector=include_vector,
                filters=convert_filters(filters) if filters else None,
                limit=top_k,
                return_properties=None,
                return_metadata=return_metadata,  # type: ignore
            )  # type: ignore
            result = [result]
        else:
            parallel_workers = [
                self.__collection.query.near_vector(  # type: ignore
                    near_vector=query_embedding,
                    distance=distance,
                    certainty=certainty,
                    include_vector=include_vector,
                    filters=convert_filters(filters) if filters else None,
                    limit=top_k,
                    return_properties=None,
                    return_metadata=return_metadata,  # type: ignore
                )
                for query_embedding in queries_embeddings
            ]
            result = await asio.gather(*parallel_workers)
        response = [self._to_document(doc) for res in result for doc in res.objects]
        return response


class IFinder:
    async def find(self, collection_name: str, **kwargs):
        logger.info(f"FINDER | collection_name=[{collection_name}]")
        store = await WeaviateDocStore.connect(
            collection_schema_name=collection_name, **kwargs
        )
        assert (
            store.get_collection_name() == collection_name.lower().capitalize()
        ), f"Mismatch in collection's settings initialization.\
                collection_name=[{collection_name}] |\
                store.collection_schema_name=[{store.__collection.name}]"

        return store


Finder = IFinder()


__all__ = ["WeaviateDocStore"]
