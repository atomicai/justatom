import asyncio as asio
import os
from pathlib import Path

import simplejson as json
from loguru import logger
from quart import Quart, request, send_from_directory, session
from quart_session import Session

from justatom.configuring.prime import Config
from justatom.etc.filters import check_filters_and_cast
from justatom.running.igni import IGNIRunner
from justatom.storing.dataset import API as DatasetApi
from justatom.storing.weaviate import Finder as WeaviateApi
from justatom.tooling.hardware import initialize_device_settings

app = Quart(
    __name__,
    static_url_path="",
    static_folder=str(Path(os.getcwd()) / "justatom" / "build"),
    template_folder=str(Path(os.getcwd()) / "justatom" / "build"),
)
app.config["SESSION_TYPE"] = "redis"
app.config["SESSION_URI"] = f'redis://{os.environ.get("REDIS_HOST", "127.0.0.1")}:6379'
logger.info(f'Session(s) storage redis=[{app.config["SESSION_URI"]}]')
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
    query, collection_name, filter_by, search_by, top_k, top_p, alpha = (
        data.get("text", "").strip(),
        data.get("collection_name", "justatom").strip(),
        data.get("filter_by", None),
        data.get("search_by", "keywords").strip(),
        data.get("top_k", 2),
        data.get("top_p", None),
        data.get("alpha", None),
    )
    session["searching"] = query  # TODO: wrap around with meta fields and prepare for logging
    if top_p is not None and search_by not in {"fusion", "atomic"}:
        msg = f"You provided `top_p`=[{top_p}] but `search_by`=[{search_by}]. Please use one of {','.join(['fusion', 'atomic'])}"
        logger.error(msg)
        return json.dumps({"msg": msg})
    filters = check_filters_and_cast(filter_by)
    store = WeaviateApi.find(
        collection_name, WEAVIATE_HOST=os.environ.get("WEAVIATE_HOST"), WEAVIATE_PORT=int(os.environ.get("WEAVIATE_PORT"))
    )
    device, _ = initialize_device_settings(use_gpus=Config.api.use_gpus, **Config.api.gpu_props)
    logger.info(f"/SEARCHING collection=[{collection_name}] device=[{device}]")
    retriever = await IGNIRunner.RETRIEVER(store=store, search_by=search_by, device=device)
    response = retriever.retrieve_topk(queries=[query], top_k=top_k, top_p=top_p, alpha=alpha, filters=filters)[0]
    return json.dumps({"docs": [d.to_dict(uuid_to_str=True) for d in response]}, ensure_ascii=False).encode("utf-8")


@app.post("/indexing")
async def index():
    data = await request.get_data(parse_form_data=True)
    data = data.decode("utf-8")
    data = json.loads(data)
    # List[str]
    collection_name, dataset_name_or_docs, index_by, batch_size = (
        data.get("collection_name", "justatom").strip(),
        data.get("dataset_name_or_docs", "justatom"),
        data.get("index_by", "keywords"),
        data.get("batch_size", 16),
    )
    docs = (
        list(DatasetApi.named(dataset_name_or_docs).iterator())
        if isinstance(dataset_name_or_docs, str)
        else [
            dict(
                content=di["content"],
                dataframe=di.get("dataframe", None),
                meta=di.get("meta", {}),
                keywords_or_phrases=di.get("keywords_or_phrases", []),
            )
            if "dataframe" in di
            else dict(
                content=di["content"],
                meta=di.get("meta", {}),
                keywords_or_phrases=di.get("keywords_or_phrases", []),
            )
            for di in dataset_name_or_docs
        ]
    )
    store = WeaviateApi.find(
        collection_name, WEAVIATE_HOST=os.environ.get("WEAVIATE_HOST"), WEAVIATE_PORT=int(os.environ.get("WEAVIATE_PORT"))
    )
    device, _ = initialize_device_settings(use_gpus=Config.api.use_gpus, **Config.api.gpu_props)
    logger.info(f"/INDEXING collection=[{collection_name}] device=[{device}]")
    indexer = await IGNIRunner.INDEXER(store=store, index_by=index_by, device=device)

    await indexer.index(docs, batch_size=int(batch_size))

    return json.dumps({"total_docs": indexer.store.count_documents()})


@app.post("/delete")
async def delete():
    data = await request.get_data(parse_form_data=True)
    data = data.decode("utf-8")
    data = json.loads(data)

    collection_name = data.get("collection_name", None)
    assert collection_name is not None, logger.error("/DELETE | `collection_name` is not specified")

    store = WeaviateApi.find(
        collection_name, WEAVIATE_HOST=os.environ.get("WEAVIATE_HOST"), WEAVIATE_PORT=int(os.environ.get("WEAVIATE_PORT"))
    )

    total_docs = store.count_documents()
    store.delete_all_documents()

    return json.dumps({"deleted_docs": total_docs})


@app.post("/patching")
async def patch():
    # `collection_name` - old collection
    # `new_collection_name` - new collection
    data = await request.get_data(parse_form_data=True)
    data = data.decode("utf-8")
    data = json.loads(data)

    collection_name = data.get("collection_name", None)
    assert collection_name is not None, logger.error("/PATCHING | `collection_name` to delete is not specified")
    new_collection_name = data.get("new_collection_name", None)
    assert new_collection_name is not None, logger.error("/PATCHING | `new_collection_name` to create is not specified")

    keep_previous_collection = data.get("keep_previous_collection", True)
    batch_size = data.get("batch_size", 256)
    patcher = await IGNIRunner.PATCHER(collection_name=collection_name, new_collection_name=new_collection_name)
    response, status = await patcher.patch(batch_size=batch_size, keep_previous_collection=keep_previous_collection)

    return json.dumps(response)


@app.post("/deletebyids")
async def deletebyids():
    data = await request.get_data(parse_form_data=True)
    data = data.decode("utf-8")
    data = json.loads(data)
    collection_name = data.get("collection_name", None)
    assert collection_name is not None, logger.error("/DELETEBYIDS | `collection_name` is not specified")
    document_ids = data.get("document_ids", None)
    assert document_ids is not None, logger.error("/DELETEBYIDS | `document_ids` are not specified")

    store = WeaviateApi.find(collection_name)
    js_existed_documents = list(store.get_document_by_ids(document_ids=document_ids, include_vector=False))
    logger.info(f"/DELETEBYIDS | Found K=[{len(js_existed_documents)}]")
    is_ok: bool = store.delete_documents(document_ids=document_ids)

    return json.dumps(
        {"deleted_docs": [js_doc.to_dict(uuid_to_str=True) for js_doc in js_existed_documents], "status": is_ok}, ensure_ascii=False
    ).encode("utf-8")


@app.post("/findbyids")
async def findbyids():
    data = await request.get_data(parse_form_data=True)
    data = data.decode("utf-8")
    data = json.loads(data)

    collection_name = data.get("collection_name", None)
    assert collection_name is not None, logger.error("/FINDBYIDS | `collection_name` is not specified")
    document_ids = data.get("document_ids", None)
    assert document_ids is not None, "/FINDBYIDS | `document_ids` are not specified"
    include_vector = data.get("include_vector", False)

    store = WeaviateApi.find(collection_name)
    response = list(store.get_document_by_ids(document_ids=document_ids, include_vector=include_vector))

    return json.dumps({"found_docs": [d.to_dict(uuid_to_str=True) for d in response]}, ensure_ascii=False).encode("utf-8")


@app.post("/find")
async def find():
    data = await request.get_data(parse_form_data=True)
    data = data.decode("utf-8")
    data = json.loads(data)

    collection_name = data.get("collection_name", None)
    include_vector = data.get("include_vector", None) or False
    assert collection_name is not None, "Collection is not specified."
    store = WeaviateApi.find(collection_name)

    result = [store._to_document(doc).to_dict(uuid_to_str=True) for doc in store.get_all_documents(include_vector=include_vector)]
    return json.dumps({"found_docs": result}, ensure_ascii=False).encode("utf-8")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5555)
