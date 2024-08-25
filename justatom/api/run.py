import os
from pathlib import Path

import simplejson as json
from loguru import logger
from quart import Quart, request, session
from quart_session import Session

from justatom.running.indexer import API as IndexerApi
from justatom.running.retriever import API as RetrieverApi
from justatom.storing.dataset import API as DatasetApi
from justatom.storing.weaviate import Finder as StoreFinder

app = Quart(
    __name__,
    static_url_path="",
    static_folder=str(Path(os.getcwd()) / "justatom" / "build" / "static"),
    template_folder=str(Path(os.getcwd()) / "justatom" / "build"),
)
app.config["SESSION_TYPE"] = "redis"
Session(app)


@app.post("/echo")
async def echo():
    data = await request.get_json()
    return {"input": data, "extra": True}


@app.post("/searching")
async def search():
    data = await request.get_data(parse_form_data=True)
    data = data.decode("utf-8")
    data = json.loads(data)
    # Placeholder to retrieve all the document(s) from the indexing stage
    logger.info(data)
    query, collection_name, search_by, top_k = (
        data.get("text", "").strip(),
        data.get("collection_name", "Default").strip(),
        data.get("search_by", "keywords").strip(),
        data.get("top_k", 2),
    )
    store = StoreFinder.find(collection_name)
    retriever = RetrieverApi.named(search_by, store=store)
    session["query"] = query
    response = retriever.retrieve_topk(queries=[query], top_k=top_k)
    logger.info(response)
    return json.dumps(
        {
            "docs": [
                dict(
                    content=di.content,
                    content_type=di.content_type,
                    score=di.score,
                )
                for di in response
            ]
        },
        ensure_ascii=False,
    ).encode("utf-8")


@app.post("/indexing")
async def index():
    data = await request.get_data(parse_form_data=True)
    data = data.decode("utf-8")
    data = json.loads(data)
    collection_name, dataset_name_or_url, index_by = (
        data.get("collection_name", "Default").strip(),
        data.get("dataset_name", "justatom"),
        data.get("index_by", "keywords"),
    )
    store = StoreFinder.find(collection_name)
    indexer = IndexerApi.named(index_by, store=store)
    docs = list(DatasetApi.named(dataset_name_or_url).iterator())

    await indexer.index(docs)

    return json.dumps({"total_docs": store.count_documents()})
