<p align="center">
    <img src="./docs/Logo.png">
</p>


Information retrieval - <a href="https://en.wikipedia.org/wiki/Information_retrieval">IR</a> is one of the most "in-demand" subtasks. In many search systems the `IR` part is implemented as a separate component `R` that narrows down the search from billions to, let's say, `k=5` candidates.

---

> Given the `K` and the collection of question-answer pairs `C`, «how good the retrieval system is» could be defined mathematically as 

```math
\varphi(R)_{K}=\sum_{i=1}^{|C|}\frac{R_{K}(q_i)\cap\{c_i\}}{|C|}
```

> Basically, for each query $q_i$ we find closest $top_k$ paragraphs and see whether the correct one - $c_i$ has been retrieved.

The good-old-fashioned solution is to use the well-known <a href="https://en.wikipedia.org/wiki/Elasticsearch">Elasticsearch</a>, which is fast and performant. However, it lacks the “smart” understanding of what you’ve asked it to search for, because under the hood it uses an inverted index algorithm (BM25). As a result, it doesn’t distinguish between “apple fruits” and “apple stocks.”

---

### Get started
> They say a good example is worth 2077 pages of API documentation, a million directives, or a thousand words.

Let's start from observing the built-in dataset - `polaroids.ai`. It consists of various snippets from movies, games and books. Below are a few examples:

<small>

|  queries (list[str])  |     content: str     |   chunk_id: str   |
|:---------------------:|:--------------------:|:-----------------:|
| 1. ...thinking about 'The Hunger Games' mechanics, if you were in the same shoes as Gale, entering your name forty-two times to feed your fam, how would you strategize your game in the actual Arena? Would you team up or go solo based on these high stakes? <br><br>2. In the universe of 'The Hunger Games', what are tesserae and what do they offer to the participants in the Harvest?    | And here's where the real interest begins. Suppose you're poor and starving. Then you can ask to be included in the Harvest more times than you're entitled to, and in return you'd get tesserae. They give you grain and oil for a whole year for one tessera per person. You won't be full, but it's better than nothing. You can take tesserae for the whole family. When I was twelve, my name was entered four times. Once by law, and once more for tesserae for Prim, my mother, and myself. The next years had to do the same. And since the price of a tessera increases by one entry each year, now that I've turned sixteen, my name will be on twenty cards. Gale is eighteen, and he's been feeding a family of five for seven years. His name will be entered forty two times! It's clear that people like Madge, who has never had to risk because of tesserae, annoy Gale. Next to us, the inhabitants of the slag heap, she simply has no chance of getting into the games. Well, almost no chance. Of course, the rules are set by the Capitol, not the districts, let alone Madge's relatives, and it's still hard to sympathize with those who, like you, don't have to trade their own skin for a piece of bread.  | 80504cd8-9b21-514c-b001-4761d8c71044         |
|-----------------------|----------------------|-------------------|
| 1. In 'Harry Potter and the Philosopher's Stone', what misconception had Harry and Hermione initially had about Snape's intentions before learning the truth? <br><br>2. Hey peeps, why is Harry all jittery and pacing around the room even after telling Hermione about the whole Snape and Voldemort situation?        | Ron was asleep in the common room - apparently, he had been waiting for their return and had dozed off unnoticed. When Harry roughly shook him, Ron began to yell something about breaking the rules of a game, as if he were dreaming about a Quidditch match. However, after a few seconds, Ron completely woke up and, with his eyes wide open, listened to the story of Hermione and Harry. Harry was so excited that he could not sit still and paced back and forth across the room, trying to stay as close to the fireplace as possible. He was still shaking with cold. 'Snape wants to steal the stone for Voldemort. And Voldemort is waiting in the forest... And all this time we thought Snape wanted to steal the stone to become rich... And Voldemort...'  | 5ad25a92-28d9-5971-a81b-4f795898eeab         |
|-----------------------|----------------------|-------------------|
| 1. Hey fellow gamers, in The Hunger Games universe, if you were in a match where your ally was taken down first like Rue, how would you strategize your next move to survive against top opponents like Cato?<br><br> 2. In the 'Hunger Games' novel, why does Cato decide to spare Katniss's life after their encounter?    | What was she babbling about? You're Rue's ally? - I... I... we teamed up. We blew up the food of the Pros. I wanted to save her. Really did. But he found her first, the guy from District One - I say. Perhaps if Cato knows I helped Rue, he will kill me quickly and painlessly. - Did you kill him? - he asks grimly. - Yes. I killed him. And I covered her body with flowers. I sang to her till she fell asleep. Tears well up in my eyes. Will and strength are leaving me. There's only Rue, the pain in my head, fear of Cato and the moan of the dying girl. - Fell asleep? - mocks Cato. - Died. I sang to her till she died - I say. - Your district... sent me bread. I raise my hand - not for an arrow; I won't have time anyway. I just blow my nose. - Cato, make it quick, okay? His face shows conflicting emotions. Cato puts down the rock and says with almost a reproach: - This time, only this time, I'm letting you go. For the girl. We are even. No one owes anything to anyone anymore, understand? I nod, because I do understand. Understand about debts. About how bad it is to have them. Understand that if Cato wins, he will return to a district that has forgotten the rules to thank me. And Cato is neglecting them, too. Right now, he's not going to crack my head with a stone.  | b317200c-7fd3-5804-bbe4-bff33432ad0e         |
|-----------------------|----------------------|-------------------|

</small>

> 🚀 We use <a href="https://github.com/pola-rs/polars">Polars</a> instead of Pandas. It’s much faster and more convenient for working with truly large datasets.
<details style="
  background-color: #000; 
  color: #fff; 
  padding: 10px; 
  border-radius: 5px;
">
<summary style="display: inline-block; 
    display: inline-block; 
    padding: 6px 12px; 
    margin-bottom: 4px;
    border: 1px solid #ccc; 
    border-radius: 4px; 
    background-color: cyan; 
    color: #fff;
    cursor: pointer;
  ">Code</summary>
<pre style="background-color: #000; color: #fff; border: none;">

```python
from pathlib import Path
from justatom.tooling.dataset import DatasetRecordAdapter

dataset_path = Path.cwd() / ".data" / "polaroids.ai.data.json"
adapter = DatasetRecordAdapter.from_source(dataset_path, lazy=True)
js_docs = list(adapter.iterator())
```
</pre>
</details>

### Datasets 📦

> Load large datasets lazily and map custom columns to `Document` schema with one adapter.

Supported input formats (via `justatom.storing.dataset.API`):
- `json`
- `jsonl`
- `parquet`
- `csv`
- `xlsx`

`DatasetRecordAdapter` maps your source columns to canonical `Document` fields:
- `content_col` -> `content`
- `queries_col` -> `meta.labels`
- `chunk_id_col` -> `id` (if provided)

All non-canonical source fields are stored in `meta` (for example: `instruction`, `input`, custom business fields).

```python
from pathlib import Path
from justatom.tooling.dataset import DatasetRecordAdapter

dataset_path = Path.home() / "IDataset" / "electrical_engineering_ru.parquet"

adapter = DatasetRecordAdapter.from_source(
  dataset_path,
  lazy=True,
  content_col="output",   # map source output -> Document.content
  queries_col="input",    # map source input -> Document.meta.labels
)

first_doc = next(adapter.iterator())
print(first_doc["content"])                  # canonical content
print(first_doc["meta"]["labels"])          # normalized labels from input
print(first_doc["meta"]["instruction"])     # extra source field moved to meta
```

> See practical notebook example: `notebook/datasets-tutorials/pipeline-large-datasets.ipynb`

### Encoders

> See <a href="https://huggingface.co/intfloat/multilingual-e5-large">E5 large</a> , <a href="https://huggingface.co/intfloat/multilingual-e5-base">E5 base</a>, <a href="https://huggingface.co/intfloat/multilingual-e5-small">E5 small</a> family of encoder models. 📎 <a href="https://arxiv.org/abs/2212.03533">paper</a>. For Russian only domain consider also <a href="https://huggingface.co/deepvk/USER-bge-m3">USER-bge-m3</a> as well. 

<details style="
  background-color: #000; 
  color: #fff; 
  padding: 10px; 
  border-radius: 5px;
">
<summary style="display: inline-block; 
    display: inline-block; 
    padding: 6px 12px; 
    margin-bottom: 4px;
    border: 1px solid #ccc; 
    border-radius: 4px; 
    background-color: cyan; 
    color: #fff;
    cursor: pointer;
  ">Code</summary>
<pre style="background-color: #000; color: #fff; border: none;">

```python
import torch
from justatom.modeling.mask import ILanguageModel
from justatom.running.encoders import EncoderRunner
from justatom.processing import RuntimeProcessor, ITokenizer

def maybe_cuda_or_mps():
    if torch.backends.mps.is_built():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda:0"
    else:
        return "cpu"

model_name_or_path = "intfloat/multilingual-e5-small"

lm_model = ILanguageModel.load(model_name_or_path) # load encoder
device = maybe_cuda_or_mps() # pick the device
runner = EncoderRunner(model=lm_model, prediction_heads=[], device=device) # create `Runner`
processor = RuntimeProcessor(ITokenizer.from_pretrained(model_name_or_path)) # load `Processor`
```
</pre>
</details>

❗️According to the <a href="https://arxiv.org/abs/2212.03533">paper</a> E5 family is trained in an asymmetric way, meaning:

> Use `"query: "` and `"passage: "` respectively for asymmetric tasks such as passage retrieval in open QA, ad-hoc information retrieval.

> Use `"query: "` prefix for symmetric tasks such as semantic similarity, bitext mining, paraphrase retrieval.

> Use `"query: "` prefix if you want to use embeddings as features, such as linear probing classification, clustering.

```python
processor.prefix = "passage: " # For indexing. For searching paste "query: "
```

> There are many libraries available to help you build an <a href="https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm">ANN</a> search solution. However, when it comes to the nitty-gritty details—like “hybrid search,” multi-vector support, and async-friendly clients—only a few solutions meet our requirements. We recommend using <a href="https://github.com/weaviate/weaviate">Weaviate</a>, which is open source and fully supports all the features mentioned above.

### Index

<details style="
background-color: #000; 
color: #fff; 
padding: 10px; 
border-radius: 5px;
">
<summary style="display: inline-block; 
display: inline-block; 
padding: 6px 12px; 
margin-bottom: 4px;
border: 1px solid #ccc; 
border-radius: 4px; 
background-color: orange; 
color: #fff;
cursor: pointer;
">Start weaviate</summary>
<p>

> ❗️By default weaviate will listen on port `2211`

```bash
docker-compose up -d
```

</p>
</details>

---

> 🔖 You can initialize the connection with a single line of code. First, pick your collection_name—think of it as a table in a standard database structure. From then on, all operations will be performed directly on that table.


<details style="
background-color: #000; 
color: #fff; 
padding: 10px; 
border-radius: 5px;
">
<summary style="display: inline-block; 
display: inline-block; 
padding: 6px 12px; 
margin-bottom: 4px;
border: 1px solid #ccc; 
border-radius: 4px; 
background-color: orange; 
color: #fff;
cursor: pointer;
">Connect to the store</summary>

<p>

```python
from justatom.storing.weaviate import Finder as WeaviateApi

collection_name = "justatom"
weaviate_host, weaviate_port = "localhost", 2211

store = await WeaviateApi.find(collection_name, WEAVIATE_HOST=weaviate_host, WEAVIATE_PORT=weaviate_port)
```

</p>
</details>

---

> ✔️ Let's double-check that everything worked out fine.

```python
n_docs = await store.count_documents()
print(n_docs)
```

>❓Are you still here? Congratulations, most of the work is almost done. Now, simply wrap the IndexerAPI pipeline and start indexing!

> Now let's create an indexing pipeline in one line

<details style="
background-color: #000; 
color: #fff; 
padding: 10px; 
border-radius: 5px;
">
<summary style="display: inline-block; 
display: inline-block; 
padding: 6px 12px; 
margin-bottom: 4px;
border: 1px solid #ccc; 
border-radius: 4px; 
background-color: red; 
color: #fff;
cursor: pointer;
">Index</summary>
<p>

```python
from justatom.running.indexer import API as IndexerAPI
# 1. "embedding" is the way to index the given ANN store (weaviate)
# 2. runner is responsible for mapping docs to embeddings
# 3. processor is responsible for tokenizing given chunks
# 4. device - compute everything on selected `device`

ix_runner = IndexerAPI.named("embedding", runner=runner, store=store, processor=processor, device=device)

async for js_batch_docs in ix_runner.index(js_docs, batch_size=64, batch_size_per_request=32):
    print(f"Done {len(js_batch_docs)} / {len(js_docs)}")

```

- `batch_size` indicates how many embeddings are processed on your device at once.
- `batch_size_per_request` specifies how many POST requests are sent simultaneously to the <a href="https://weaviate.io/developers/weaviate/client-libraries/python/async">Weaviate</a> ANN document store.
</p>
</details>

### Search
> For the search API we implement various algorithms including:
- search via native `bm25`: $R_k$
- semantic search (by embedding): $R_e$
- hybrid search $\alpha \in [0.0, \dots ,1.0]$. $(1 - \alpha) \times R_k + \alpha \times R_e$
- fusion retriever (coming soon).

<details style="
background-color: #000; 
color: #fff; 
padding: 10px; 
border-radius: 5px;
">
<summary style="display: inline-block; 
display: inline-block; 
padding: 6px 12px; 
margin-bottom: 4px;
border: 1px solid #ccc; 
border-radius: 4px; 
background-color: red; 
color: #fff;
cursor: pointer;
">Search</summary>
<p>

> ❗️According to the <a href="https://arxiv.org/abs/2212.03533">paper</a> E5 family is trained in an asymmetric way, meaning we have to set `prefix` back to `query: `

```python
processor.prefix = "query: "
```

> Now, let's create some queries we would like to search for:

```python
queries = [
    "thinking about 'The Hunger Games' mechanics, if you were in the same shoes as Gale, entering your name forty-two times to feed your fam, how would you strategize your game in the actual Arena? Would you team up or go solo based on these high stakes?",
    "In the universe of 'The Hunger Games', what are tesserae and what do they offer to the participants in the Harvest?",
    "In 'Harry Potter and the Philosopher's Stone', what misconception had Harry and Hermione initially had about Snape's intentions before learning the truth?",
    "Hey peeps, why is Harry all jittery and pacing around the room even after telling Hermione about the whole Snape and Voldemort situation?",
    "Hey fellow gamers, in The Hunger Games universe, if you were in a match where your ally was taken down first like Rue, how would you strategize your next move to survive against top opponents like Cato?",
    "In the 'Hunger Games' novel, why does Cato decide to spare Katniss's life after their encounter?"
]
```

> Pipeline below uses the following algorithms:
- `bm25`
- `embedding`
- `hybrid`

```python
from justatom.running.retriever import API as RetrieverApi

R_k = RetrieverApi.named("keywords", store=store) # Retriever by keywords
R_e = RetrieverApi.named("embedding", store=store, runner=runner, processor=processor) # Retriever by embedding
R_h = RetrieverApi.named("hybrid", store=store, runner=runner, processor=processor) # Retriever via `hybrid` search

for i, query in enumerate(queries):
    R_k_topk = (await R_k.retrieve_topk(query, top_k=1))[0]
    print(f"KEYWORDS | {R_k_topk}")
    R_e_topk = (await R_e.retrieve_topk(query, top_k=1))[0]
    print(f"EMBEDDING | {R_e_topk}")
    alpha = 0.78
    R_h_topk = (await R_h.retrieve_topk(query, top_k=1, alpha=alpha))[0] # Let's try alpha=0.78
    print(f"HYBRID | alpha={alpha} | {R_h_topk}")
    print("\n")
```


</p>
</details>


### License

MIT license. Free for commercial use

### Feedback
> Contact me via: 
- 🔗 <a href="https://www.linkedin.com/in/itarlinskiy/">linkedin</a>
- ✈️ <a href="https://t.me/itarlinskiy">tg</a>
- 👥 <a href="https://vk.com/itarlinskiy">vk</a>
or better visit <a href="justatom.ai">blog</a> and see many more posts with direct contact link.
