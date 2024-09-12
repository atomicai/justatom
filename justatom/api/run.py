import os
from pathlib import Path

import simplejson as json
from loguru import logger
from quart import Quart, request, session
from quart_session import Session

from justatom.etc.filters import check_filters_and_cast
from justatom.running.igni import IGNIRunner
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
    query, collection_name, keywords, filter_by, search_by, top_k = (
        data.get("text", "").strip(),
        data.get("collection_name", "Default").strip(),
        data.get("keywords", None),
        data.get("filter_by", None),
        data.get("search_by", "keywords").strip(),
        data.get("top_k", 2),
    )
    store = StoreFinder.find(collection_name)
    retriever = IGNIRunner.RETRIEVER(store=store, search_by=search_by, prefix_to_use="query:")
    session["searching"] = query  # TODO: wrap around with meta fields and prepare for logging
    filters = check_filters_and_cast(filter_by)
    response = retriever.retrieve_topk(queries=[query], top_k=top_k, filters=filters, keywords=keywords)
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
    collection_name, dataset_name_or_docs, index_by, batch_size = (
        data.get("collection_name", "Default").strip(),
        data.get("dataset_name_or_docs", "justatom"),
        data.get("index_by", "keywords"),
        data.get("batch_size", 16),
    )
    store = StoreFinder.find(collection_name)
    indexer = IGNIRunner.INDEXER(store=store, index_by=index_by, prefix_to_use="passage:")
    session["indexing"] = dataset_name_or_docs  # TODO: wrap around with meta fields and prepare for logging
    docs = (
        list(DatasetApi.named(dataset_name_or_docs).iterator())
        if isinstance(dataset_name_or_docs, str)
        else [
            dict(
                content=di["content"],
                dataframe=di.get("dataframe", None),
                keywords=di.get("keywords", None),
            )
            if "dataframe" in di
            else dict(content=di["content"], dataframe=di["title"], keywords=di["keywords"])
            for di in dataset_name_or_docs
        ]
    )

    await indexer.index(docs, batch_size=int(batch_size), device=indexer.device)

    return json.dumps({"total_docs": store.count_documents()})
