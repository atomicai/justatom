from typing import List, Union
from quart import Quart, request, session
from quart_session import Session
import simplejson as json
from justatom.storing.weaviate import Finder as StoreFinder
from justatom.running.retriever import Finder as RetrieverFinder
from loguru import logger

from pathlib import Path
import os


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
    retriever = RetrieverFinder.find(search_by, store=store)
    session["query"] = query
    response = retriever.retrieve_topk(queries=[query], top_k=top_k)
    logger.info(response)
    return json.dumps({"docs": response}, ensure_ascii=False).encode("utf-8")
