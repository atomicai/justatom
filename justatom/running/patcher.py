from loguru import logger
from more_itertools import chunked
from tqdm import tqdm

from justatom.running.mask import IPatcherRunner
from justatom.storing.weaviate import Finder as WeaviateApi
from justatom.storing.weaviate import WeaviateDocStore
from justatom.tooling.stl import AsyncConstructor


class PatcherRunner(IPatcherRunner, AsyncConstructor):
    async def __init__(self, collection_name, new_collection_name):
        super().__init__()
        self.collection_name = collection_name
        self.new_collection_name = new_collection_name
        self.current_doc_store: WeaviateDocStore = await WeaviateApi.find(collection_name)
        self.next_doc_store: WeaviateDocStore = await WeaviateApi.find(new_collection_name)

    async def patch(self, batch_size: int = 128, **kwargs) -> tuple:
        it_collection_data = await self.current_doc_store.get_all_documents(include_vector=True)
        total_written_docs, status = 0, 200
        for batch_idx, js_batch_we_docs in tqdm(enumerate(chunked(it_collection_data, n=batch_size))):
            js_batch_docs = [self.current_doc_store._to_document(js_we_doc) for js_we_doc in js_batch_we_docs]
            try:
                batch_written_docs = await self.next_doc_store.write_documents(js_batch_docs)
            except:  # noqa
                logger.info(f"{self.__class__.__name__} | `batch_idx`=[{batch_idx}]")
                status = 404
            else:  # TODO:
                total_written_docs += batch_written_docs
                logger.info(
                    f"""
                            {self.__class__.__name__} | count({self.new_collection_name})=[{total_written_docs}]
                    """
                )
        is_removal_ok: bool = await self.current_doc_store.delete_all_documents()
        if is_removal_ok:
            msg = f"{self.__class__.__name__} | Removal on `collection_name`=[{self.collection_name}]  "
            logger.info(msg)
        else:
            msg, status = f"Remove on `collection_name`=[{self.collection_name}] failed", 404
            logger.error(msg)
        n_count_docs_current = await self.current_doc_store.count_documents()
        n_count_docs_next = await self.next_doc_store.count_documents()
        return {
            f"count[{self.collection_name}]": n_count_docs_current,
            f"count[{self.new_collection_name}]": n_count_docs_next,
        }, status
