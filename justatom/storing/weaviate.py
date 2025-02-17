import base64
import datetime
import json
import os
from collections.abc import Generator
from dataclasses import asdict
from typing import Any

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
from justatom.etc.pattern import singleton
from justatom.etc.schema import Document
from justatom.etc.serialization import default_from_dict, default_to_dict
from justatom.etc.types import DuplicatePolicy, SearchPolicy
from justatom.tooling.stl import merge_in_order

# See https://weaviate.io/developers/weaviate/config-refs/datatypes#:~:text=DataType%3A%20object%20%E2%80%8B&text=The%20object%20type%20allows%20you,be%20nested%20to%20any%20depth.&text=As%20of%201.22%20%2C%20object%20and,not%20indexed%20and%20not%20vectorized.

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


class WeaviateDocStore:
    """
    WeaviateDocumentStore is a Document Store for Weaviate.
    It can be used with Weaviate Cloud Services or self-hosted instances.

    Usage example with Weaviate Cloud Services:
    ```python
    import os
    from haystack_integrations.document_stores.weaviate.auth import AuthApiKey
    from haystack_integrations.document_stores.weaviate.document_store import WeaviateDocumentStore

    os.environ["WEAVIATE_API_KEY"] = "MY_API_KEY

    document_store = WeaviateDocumentStore(
        url="rAnD0mD1g1t5.something.weaviate.cloud",
        auth_client_secret=AuthApiKey(),
    )
    ```

    Usage example with self-hosted Weaviate:
    ```python
    from haystack_integrations.document_stores.weaviate.document_store import WeaviateDocumentStore

    document_store = WeaviateDocumentStore(url="http://localhost:8080")
    ```
    """

    def __init__(
        self,
        *,
        url: str | None = None,
        collection_name: str | None = "Default",
        collection_settings: dict[str, Any] | None = None,
        auth_client_secret: AuthCredentials | None = None,
        additional_headers: dict | None = None,
        embedded_options: EmbeddedOptions | None = None,
        additional_config: AdditionalConfig | None = None,
        grpc_port: int = 50051,
        grpc_secure: bool = False,
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
        self._client = weaviate.WeaviateClient(
            connection_params=(
                weaviate.connect.base.ConnectionParams.from_url(url=url, grpc_port=grpc_port, grpc_secure=grpc_secure)
                if url
                else None
            ),
            auth_client_secret=(auth_client_secret.resolve_value() if auth_client_secret else None),
            additional_config=additional_config,
            additional_headers=additional_headers,
            embedded_options=embedded_options,
            skip_init_checks=False,
        )
        self._client.connect()

        # Test connection, it will raise an exception if it fails.
        self._client.collections._get_all(simple=True)
        # TODO: Re=parametrize to make it friendly for hybrid-search via bm25 + embedding search.
        if collection_settings is None:
            collection_settings = {
                "class": collection_name.capitalize(),
                "invertedIndexConfig": {"indexNullState": True},
                "properties": DOCUMENT_COLLECTION_PROPERTIES,
            }
        else:
            # Set the class if not set
            collection_settings = merge_in_order(collection_settings, {"class": collection_name.capitalize()})
            # Set the properties if they're not set
            collection_settings["properties"] = collection_settings.get("properties", DOCUMENT_COLLECTION_PROPERTIES)

        if not self._client.collections.exists(collection_settings["class"]):
            self._client.collections.create_from_dict(collection_settings)

        self._url = url
        self._collection_settings = collection_settings
        self._auth_client_secret = auth_client_secret
        self._additional_headers = additional_headers
        self._embedded_options = embedded_options
        self._additional_config = additional_config
        self._collection = self._client.collections.get(collection_settings["class"])

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
            collection_settings=self._collection_settings,
            auth_client_secret=(self._auth_client_secret.to_dict() if self._auth_client_secret else None),
            additional_headers=self._additional_headers,
            embedded_options=embedded_options,
            additional_config=additional_config,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WeaviateDocStore":
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

    def count_documents(self) -> int:
        """
        Returns the number of documents present in the DocumentStore.
        """
        total = self._collection.aggregate.over_all(total_count=True).total_count
        return total if total else 0

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
            logger.warning(f"[meta={data['meta']}] is present and will be ignored since it is NOT registred in a collection.")
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
            if weaviate_meta.score is not None:
                document_data["score"] = weaviate_meta.score
            elif weaviate_meta.certainty is not None:
                document_data["score"] = weaviate_meta.certainty

        return Document.from_dict(document_data)

    def _check_keywords(self, docs: list[Document], keywords: list[str] | None = None):
        response = docs
        if keywords:
            response = [doc for doc in response if any([kw in doc.keywords for kw in keywords])]
        return response

    def _query(self) -> list[dict[str, Any]]:
        # properties = [p.name for p in self._collection.config.get().properties]
        try:
            result = self._collection.iterator(include_vector=True, return_properties=None)
        except weaviate.exceptions.WeaviateQueryError as e:
            msg = f"Failed to query documents in Weaviate. Error: {e.message}"
            raise DocumentStoreError(msg) from e
        return result

    def _query_with_filters(self, filters: dict[str, Any]) -> list[dict[str, Any]]:
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
        offset = 0
        partial_result = None
        result = []
        # Keep querying until we get all documents matching the filters
        while partial_result is None or len(partial_result.objects) == DEFAULT_QUERY_LIMIT:
            try:
                partial_result = self._collection.query.fetch_objects(
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

    def filter_documents(self, filters: dict[str, Any] | None = None) -> list[Document]:
        """
        Returns the documents that match the filters provided.

        For a detailed specification of the filters, refer to the
        DocumentStore.filter_documents() protocol documentation.

        :param filters: The filters to apply to the document list.
        :returns: A list of Documents that match the given filters.
        """
        result = []
        if filters:  # noqa: SIM108
            result = self._query_with_filters(filters)
        else:
            result = self._query()
        return [self._to_document(doc) for doc in result]

    def _batch_write(self, documents: list[Document]) -> int:
        """
        Writes document to Weaviate in batches.
        Documents with the same id will be overwritten.
        Raises in case of errors.
        """

        with self._client.batch.dynamic() as batch:
            for doc in documents:
                if not isinstance(doc, Document):
                    msg = f"Expected a Document, got '{type(doc)}' instead."
                    raise ValueError(msg)

                batch.add_object(
                    properties=self._to_data_object(doc),
                    collection=self._collection.name,
                    uuid=generate_uuid5(doc.id),
                    vector=doc.embedding,
                )
        if failed_objects := self._client.batch.failed_objects:
            # We fallback to use the UUID if the _original_id is not present, this is just to be
            mapped_objects = {}
            for obj in failed_objects:
                properties = obj.object_.properties or {}
                # We get the object uuid just in case the _original_id is not present.
                # That's extremely unlikely to happen but let's stay on the safe side.
                id_ = properties.get("_original_id", obj.object_.uuid)
                mapped_objects[id_] = obj.message

            msg = "\n".join(
                [f"Failed to write object with id '{id_}'. Error: '{message}'" for id_, message in mapped_objects.items()]
            )
            raise DocumentStoreError(msg)

        # If the document already exists we get no status message back from Weaviate.
        # So we assume that all Documents were written.
        return len(documents)

    def _write(self, documents: list[Document], policy: DuplicatePolicy) -> int:
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

            if policy == DuplicatePolicy.SKIP and self._collection.data.exists(uuid=generate_uuid5(doc.id)):
                # This Document already exists, we skip it
                continue

            try:
                self._collection.data.insert(
                    uuid=generate_uuid5(doc.id),
                    properties=self._to_data_object(doc),
                    vector=doc.embedding,
                )

                written += 1
            except weaviate.exceptions.UnexpectedStatusCodeError:
                if policy == DuplicatePolicy.FAIL:
                    duplicate_errors_ids.append(doc.id)
        if duplicate_errors_ids:
            msg = f"IDs '{', '.join(duplicate_errors_ids)}' already exist in the document store."
            raise DuplicateDocumentError(msg)
        return written

    def write_documents(self, documents: list[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE) -> int:
        """
        Writes documents to Weaviate using the specified policy.
        We recommend using a OVERWRITE policy as it's faster than other policies for Weaviate since it uses
        the batch API.
        We can't use the batch API for other policies as it doesn't return any information whether the document
        already exists or not. That prevents us from returning errors when using the FAIL policy or skipping a
        Document when using the SKIP policy.
        """
        if policy in [DuplicatePolicy.NONE, DuplicatePolicy.OVERWRITE]:
            return self._batch_write(documents)

        return self._write(documents, policy)

    def get_all_documents(self, include_vector: bool = False) -> Generator:
        props = dict(include_vector=include_vector)
        for obj in self._collection.iterator(**props):  # noqa: UP028
            yield obj

    def get_document_by_id(self):
        pass

    def delete_documents(self, document_ids: list[str]) -> None:
        """
        Deletes all documents with matching document_ids from the DocumentStore.

        :param document_ids: The object_ids to delete.
        """
        weaviate_ids = [generate_uuid5(doc_id) for doc_id in document_ids]
        self._collection.data.delete_many(where=weaviate.classes.query.Filter.by_id().contains_any(weaviate_ids))

    def delete_all_documents(self) -> bool:
        it = self.get_all_documents()
        ids = [x.properties["_original_id"] for x in it]
        if len(ids) > 0:
            try:
                self.delete_documents(document_ids=ids)
            except:  # noqa: E722
                logger.error(f"Error deleting documents for {self._collection_settings.get('class')}, see logs for more details.")
                return False
            else:
                return True
        else:
            logger.info(f"Nothing to delete in {self._collection_settings.get('class')}")

    def search_by_keywords(
        self,
        query: str,
        policy: SearchPolicy | None = SearchPolicy.BM25,
        filters: dict[str, Any] | None = None,
        keywords: list | None = None,
        top_k: int | None = None,
        include_vector: bool | None = False,
    ) -> list[Document]:
        # properties = [p.name for p in self._collection.config.get().properties]
        if policy == SearchPolicy.BM25:
            result = self._collection.query.bm25(
                query=query,
                filters=convert_filters(filters) if filters else None,
                limit=top_k,
                include_vector=include_vector,
                query_properties=["content"],
                return_properties=None,
                return_metadata=MetadataQuery(distance=True, score=True, explain_score=True, certainty=True),
            )
        else:
            msg = f"You specified {str(policy)} that is not compatable with [search_by_keywords]. Only [BM25] is avalaible"
            logger.error(msg)
            raise ValueError(msg)
        response = [self._to_document(doc) for doc in result.objects]
        response = self._check_keywords(response, keywords=keywords)
        return response

    def search(
        self,
        query: str,
        query_embedding: list[float],
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
        )
        result = self._collection.query.hybrid(
            query=query,
            vector=query_embedding,
            alpha=alpha,
            limit=top_k,
            filters=convert_filters(filters) if filters else None,
            return_metadata=return_metadata,
            include_vector=include_vector,
            query_properties=["content"],
        )

        response = [self._to_document(doc) for doc in result.objects]
        response = self._check_keywords(response, keywords)
        return response

    def search_by_embedding(
        self,
        query_embedding: list[float],
        filters: dict[str, Any] | None = None,
        keywords: list[str] | None = None,
        top_k: int | None = None,
        distance: float | None = None,
        certainty: float | None = None,
        return_metadata: list[str] | None = None,
    ) -> list[Document]:
        if distance is not None and certainty is not None:
            msg = "Can't use 'distance' and 'certainty' parameters together"
            raise ValueError(msg)
        return_metadata = ["certainty"] if return_metadata is None else return_metadata
        # properties = [p.name for p in self._collection.config.get().properties]
        result = self._collection.query.near_vector(
            near_vector=query_embedding,
            distance=distance,
            certainty=certainty,
            include_vector=True,
            filters=convert_filters(filters) if filters else None,
            limit=top_k,
            return_properties=None,
            return_metadata=return_metadata,
        )

        response = [self._to_document(doc) for doc in result.objects]
        response = self._check_keywords(response)
        return response


@singleton
class IFinder:
    store: dict[str, WeaviateDocStore] = dict()

    def find(self, collection_name: str, **kwargs):
        if collection_name not in self.store:
            WEAVIATE_HOST = kwargs.get("WEAVIATE_HOST") or os.environ.get("WEAVIATE_HOST")
            WEAVIATE_PORT = kwargs.get("WEAVIATE_PORT") or os.environ.get("WEAVIATE_PORT")
            self.store[collection_name] = WeaviateDocStore(
                collection_name=collection_name,
                url=f"http://{WEAVIATE_HOST}:{WEAVIATE_PORT}",
            )
        return self.store[collection_name]


Finder = IFinder()


__all__ = ["WeaviateDocStore"]
