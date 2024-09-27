import asyncio as asio
import os
from pathlib import Path

import simplejson as json
from loguru import logger
from quart import Quart, request, send_from_directory, session
from quart_session import Session

from justatom.etc.filters import check_filters_and_cast
from justatom.running.igni import IGNIRunner
from justatom.storing.dataset import API as DatasetApi
from justatom.storing.weaviate import Finder as StoreFinder

app = Quart(
    __name__,
    static_url_path="",
    static_folder=str(Path(os.getcwd()) / "justatom" / "build"),
    template_folder=str(Path(os.getcwd()) / "justatom" / "build"),
)
app.config["SESSION_TYPE"] = "redis"
Session(app)


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
async def main(path):
    if path != "" and os.path.exists(app.static_folder + "/" + path):
        response = await send_from_directory(app.static_folder, path)
    else:
        response = await send_from_directory(app.static_folder, "index.html")
    return response


@app.before_serving
async def serve():
    from justatom.mq.clients.rabbitmq import RabbitMQClient
    from justatom.mq.settings.rabbitmq import SettingsRabbitMQ

    settings = SettingsRabbitMQ()

    CLIENT_NAME = "consumer"

    client = RabbitMQClient(settings, client_name=CLIENT_NAME)

    server = await IGNIRunner.SERVER()

    app.run_and_serve = asio.get_event_loop().create_task(client.consume_with_callback(callback=server, routing_key=CLIENT_NAME))


@app.after_serving
async def finish():
    loop = asio.get_event_loop()
    try:
        app.run_and_serve.cancel()
    except asio.CancelledError:
        logger.warning(f"Coulnd't stop the task with id {app.run_and_serve}")
    finally:
        loop.close()


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
    retriever = await IGNIRunner.RETRIEVER(store=store, search_by=search_by, prefix_to_use="query:")
    session["searching"] = query  # TODO: wrap around with meta fields and prepare for logging
    filters = check_filters_and_cast(filter_by)
    response = retriever.retrieve_topk(queries=[query], top_k=top_k, filters=filters, keywords=keywords)[0]
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
    indexer = await IGNIRunner.INDEXER(store=store, index_by=index_by, prefix_to_use="passage:")
    session["indexing"] = dataset_name_or_docs  # TODO: wrap around with meta fields and prepare for logging
    docs = (
        list(DatasetApi.named(dataset_name_or_docs).iterator())
        if isinstance(dataset_name_or_docs, str)
        else [
            dict(
                content=di["content"],
                dataframe=di.get("dataframe", None),
                keywords=di.get("keywords", None),
                meta=di.get("meta", {}),
            )
            if "dataframe" in di
            else dict(content=di["content"], dataframe=di["title"], keywords=di["keywords"], meta=di.get("meta", {}))
            for di in dataset_name_or_docs
        ]
    )

    await indexer.index(docs, batch_size=int(batch_size), device=indexer.device)

    return json.dumps({"total_docs": store.count_documents()})
