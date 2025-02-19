from loguru import logger
from more_itertools import chunked
from tqdm import tqdm

from justatom.running.mask import IPatcherRunner
from justatom.storing.weaviate import Finder as WeaviateApi


class PatcherRunner(IPatcherRunner):
    def __init__(self, collection_name, new_collection_name):
        self.collection_name = collection_name
        self.new_collection_name = new_collection_name
        self.st_prev = WeaviateApi.find(collection_name)
        self.st_next = WeaviateApi.find(new_collection_name)

    async def patch(self, batch_size: int = 128, **kwargs) -> tuple:
        it_collection_data = self.st_prev.get_all_documents(include_vector=True)
        for batch_idx, js_batch_we_docs in tqdm(enumerate(chunked(it_collection_data, n=batch_size))):
            js_batch_docs = [self.st_prev._to_document(js_we_doc) for js_we_doc in js_batch_we_docs]
            logger.info(f"batch_idx=[{str(batch_idx)}]")
            try:
                self.st_next.write_documents(js_batch_docs)
            except:
                return {
                    f"COUNT[{self.collection_name}]": self.st_prev.count_documents(),
                    f"COUNT[{self.new_collection_name}]": self.st_next.count_documents(),
                }, 404
            else:  # TODO:
                pass
        is_removal_ok: bool = self.st_prev.delete_all_documents()
        if is_removal_ok:
            msg = f"Removal on `collection_name`=[{self.collection_name}] is successfull"
            logger.info(msg)
        else:
            msg = f"Remove on `collection_name`=[{self.collection_name}] is wrong"
            logger.error(msg)
        return {
            f"COUNT[{self.collection_name}]": self.st_prev.count_documents(),
            f"COUNT[{self.new_collection_name}]": self.st_next.count_documents(),
        }, 200
