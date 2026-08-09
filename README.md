# Enhancing Epidemiological Intelligence

**A RAG Approach to Disease Outbreak Monitoring**


- [Enhancing Epidemiological Intelligence](#enhancing-epidemiological-intelligence)
  - [Introduction](#introduction)
  - [Getting Started](#getting-started)
    - [Python Environment](#python-environment)
    - [Database Setup](#database-setup)
    - [DB Schema](#db-schema)
    - [LangChain](#langchain)
    - [Data](#data)
      - [Acquisition \& Pre-Processing](#acquisition--pre-processing)
      - [Data Ingestion](#data-ingestion)
    - [LLM](#llm)
      - [Ollama](#ollama)
      - [Embeddings](#embeddings)
      - [Retrieval](#retrieval)
    - [UI](#ui)
      - [Advanced UI](#advanced-ui)
  - [Evaluation](#evaluation)
    - [Sample Questions](#sample-questions)
    - [Notes on Batches](#notes-on-batches)
      - [Batch 0](#batch-0)
      - [Batch 1](#batch-1)
  - [Cosine Similarity in Vector Search](#cosine-similarity-in-vector-search)
    - [What is Cosine Similarity?](#what-is-cosine-similarity)
    - [Cosine Distance](#cosine-distance)
    - [Interpreting Results](#interpreting-results)
  - [References](#references)
    - [YouTube](#youtube)
    - [Articles](#articles)
    - [Repos](#repos)
    - [Data](#data-1)
    - [Scientific Papers](#scientific-papers)
    - [Technical Articles](#technical-articles)


## Introduction

This repository contains the code for my master's thesis project. It is designed to be self-contained and can be run with or without knowledge of the paper's broader context.

The repository contains all required elements for a running RAG application, from data collection, pre-processing through storage, retrieval and generation.

The corresponding text can be found in this [Google Doc](https://docs.google.com/document/d/1yrXBIel38MnqWNlNvMyZ0F4Ly4uRXjIqJBo9NjEtcms).

## Getting Started

This RAG application consists of various elements. The setup of each is detailed below:

```mermaid
graph TD;
    A[Frontend UI] --> |Query| B[Embedding Model]
    B --> |Embedded Query| C[Vector Store]
    C --> |Retrieved Relevant Documents| D[Enhanced Query with Context]
    D --> E[LLM]
    E --> |Generated Response| A
```

### Python Environment

This project uses [uv](https://docs.astral.sh/uv/) with Python 3.12. From the repository root, create or update the local `.venv` and install the locked dependencies:

```sh
uv sync --dev
```

Run project commands through `uv` so they consistently use this environment:

```sh
uv run pytest
uv run ruff check .
```

The legacy `requirements*.txt` files describe the original thesis environment and are not the reproducible installation source. `pyproject.toml` and `uv.lock` are authoritative.

### Database Setup

This project uses the [pgvector](https://github.com/pgvector/pgvector) Postgres extension as a vector store. This allows the data to be stored alongside the embeddings and as such both can be accessed easily through any SQL querying utility.

If you do not wish to run this locally, a cloud-based service such as [Supabase](https://supabase.com/modules/vector) could also be used.

There is a [docker-compose.yml](./app/docker/docker-compose.yml) which sets up a local PGVector instance as a [Docker](https://www.docker.com/products/docker-desktop/) container. **Note:** Please create an empty subdirectory `pgvector_data` before bringing the container up for the first time. This will be mounted as a volume within the Docker container.

```sh
mkdir pgvector_data
docker compose up -d
```

For Neon or another PostgreSQL database, configure `DATABASE_URL` in the root `.env`, then apply the versioned schema migrations from the repository root:

```sh
uv run alembic upgrade head
```

The migration enables pgvector and creates persistent `ingest` staging and canonical `rag` schemas. The legacy `app/db/create_db.py` path remains only for the original local prototype tables.


### DB Schema
<!-- BEGIN_SQLALCHEMY_DOCS -->
```mermaid
erDiagram
  document {
    INTEGER id PK
    INTEGER batch "nullable"
    TEXT contents "nullable"
    TEXT summary "nullable"
    JSON meta "nullable"
    DATETIME event_date "nullable"
    TEXT url "nullable"
    DATETIME published_at "nullable"
    DATETIME created_at "nullable"
  }

  embedding {
    INTEGER id PK
    TEXT model "nullable"
    INTEGER document_id FK
    INTEGER chunk_id
    HALFVEC(256) embedding_256 "nullable"
    HALFVEC(384) embedding_384 "nullable"
    HALFVEC(512) embedding_512 "nullable"
    HALFVEC(768) embedding_768 "nullable"
    HALFVEC(1024) embedding_1024 "nullable"
    HALFVEC(1536) embedding_1536 "nullable"
    HALFVEC(3072) embedding_3072 "nullable"
    HALFVEC(4096) embedding_4096 "nullable"
    HALFVEC(8192) embedding_8192 "nullable"
  }

  v_doc_embedding {
    INTEGER document_id PK
    INTEGER embedding_id PK
    INTEGER batch "nullable"
    TEXT model "nullable"
    INTEGER chunk_id "nullable"
    TEXT contents "nullable"
    HALFVEC embedding "nullable"
    TEXT summary "nullable"
    JSON meta "nullable"
    DATETIME published_at "nullable"
    TEXT url "nullable"
  }

  country_lookup {
    INTEGER id PK
    TEXT country_code "nullable"
    TEXT country_name "nullable"
    TEXT region "nullable"
    TEXT subregion "nullable"
    DATETIME created_at "nullable"
  }

  document ||--o{ embedding : document_id

```
<!-- END_SQLALCHEMY_DOCS -->


### LangChain

This code uses [LangChain](https://python.langchain.com/docs/introduction/) to abstract away some of the lower-level interactions with our LLMs and data.

At the time of writing (Jan 2025) LangChain is quite far behind in the version of pgvector it supports (v0.2.5 – current version is v0.3.6). There is an open [PR](https://github.com/langchain-ai/langchain-postgres/pull/147) for supporting the new features (especially support for the sparse vector type `halfvec`).
This version of the code can be installed directly from GitHub:

```sh
pip install git+https://github.com/langchain-ai/langchain-postgres@c32f6beb108e37aad615ee3cbd4c6bd4a693a76d
```

### Data

A list of interesting data sources pertaining to malaria and other tropical diseases can be found in the subfolders under [./data-collection](./data-collection/). The code in this repo currently uses data scraped from [WHO DONs](./data-collection/acquisition/WHO%20DONs/) to populate our RAG knowledgebase. It should be fairly straightforward to adapt it to other sources.

#### Acquisition & Pre-Processing

See [./app/who-don-retriever](./app/who-don-retriever/) for scripts to scrape and clean the data. In this directory, you'll also find a [README](./app/who-don-retriever/README.md) outlining the process.

**Note:** At the time of writing, a non-packaged version of the Markdownify library must be installed. This has better support for tables in Markdown. Some of the DONs contain HTML tables which would otherwise be lost:

```sh
pip install git+https://github.com/matthewwithanm/python-markdownify@3026602686f9a77ba0b2e0f6e0cbd42daea978f5
```

#### Data Ingestion

The new ingestion command retrieves WHO DON records and persists immutable source observations in the `ingest` schema:

```sh
uv run python -m app.ingestion extract
```

Each invocation records counts and status in `ingest.run`. A failed run can be resumed idempotently with the run UUID:

```sh
uv run python -m app.ingestion extract --run-id <uuid>
```

Normalize a completed run into canonical `rag.document` rows:

```sh
uv run python -m app.ingestion transform --run-id <uuid>
```

Transformation is batched and resumable. Each raw record is marked `transformed` or `rejected`, and documents are upserted by `(source, source_id)`.

List the tracked chunking profiles and generate a dataset:

```sh
uv run python -m app.ingestion profiles
uv run python -m app.ingestion chunk --profile who-sections-1200
uv run python -m app.ingestion chunk --profile recursive-1000-150
```

Profiles are defined in `app/config/chunk_profiles.toml`. Each profile has a deterministic configuration hash, and each source document can retain multiple chunk datasets. Completed datasets are skipped on reruns.

Validate the configured local OpenAI-compatible embedding endpoint (for example LM Studio), then incrementally embed a profile through LangChain:

```sh
uv run python -m app.ingestion embed-check
uv run python -m app.ingestion embed --profile who-sections-1200 --limit 100
uv run python -m app.ingestion embed --profile who-sections-1200
```

`--limit` is the maximum number of new chunks for that invocation. Existing current embeddings are skipped, so the unlimited command can safely resume an interrupted or partial run. Batch size can be tuned with `--batch-size` or `EMBEDDING_BATCH_SIZE`.

Run profile-aware cosine retrieval against the current chunks and embeddings:

```sh
uv run python -m app.retrieval profiles
uv run python -m app.retrieval search --profile sections \
  "What disease outbreaks affected Uganda?"
```

Retrieval supports `--limit`, repeatable `--section`, `--source`, `--published-after`, and `--published-before` filters. Profiles are defined in `app/config/retrieval_profiles.toml`. The legacy CSV loaders remain available only for the original thesis dataset.


### LLM

All components of this application can be run locally without accessing resources in the cloud. This includes the language and embedding models.

#### Ollama

[Ollama](https://ollama.com/) makes it easy to run LLMs locally. Download and run the installer. Once installed, run your model of choice, e.g. [llama3.2 3B](https://ollama.com/library/llama3.2):

The following are some suggested options for running the model on a separate computer in the same network:

```
export OLLAMA_HOST=0.0.0.0
export OLLAMA_KEEP_ALIVE=15m
export OLLAMA_FLASH_ATTENTION=true
export OLLAMA_KV_CACHE_TYPE=q8_0
ollama serve
```

By default, Ollama will expose its API on port 11434.

Also, by default, Ollama will limit its context window to 2048 tokens. This is too low for our use case. Therefore, we should adjust it before running our model or simply create our own model version with an expanded context window. To do so:

```
ollama run llama3.2
...
>>> /set parameter num_ctx 16768
>>> /save llama3.2_16kctx
>>> /bye
...
```


#### Embeddings

By default, this app uses the [all-MiniLM-L6-v2](https://www.sbert.net/) Sentence Transformer model to generate the embeddings for our vector store. An other model which works very well for embeddings is [nomic-embed-text-v1.5](https://www.nomic.ai/blog/posts/nomic-embed-text-v1). Run the following to pull the models into Ollama:

```sh
ollama pull all-minilm
...
ollama pull nomic-embed-text
```

Once the models have been downloaded, the next step is to create the embeddings for our documents.
Run [./app/db/load_embeddings.py](./app/db/load_embeddings.py):

```sh
python load_embeddings.py
```

**Note:** This will take some time to process – expect at least 15 minutes on a modern Mac laptop.


#### Retrieval

TODO:

### UI

If you've made it this far – great! At this point, run the front-end application (from within the [app](./app/) directory):

```sh
flask run
```

You should now be able to reach a simple chat-style interface at http://127.0.0.1:5000.


![./app/static/rag-chat-ui](./app/static/rag-chat-ui.png)


#### Advanced UI

There is a more advanced UI frontend that can be run be

```sh
gradio gradio-ui.py
```

You can reach the application by pointing your browser to http://127.0.0.1:7860/


## Evaluation

### Sample Questions

- In which countries is Malaria most prevalent?
- Which diseases are prevalent in Kenya?
- Which were the largest disease outbreaks in the last 20 years?
- Where were outbreaks with the most severe impacts, e.g. deaths?


### Notes on Batches

#### Batch 0

- Baseline attempt
- DONs were put together from two fields mainly
- Embeddings loaded for nomic and embed-all
- I think this data is now stored in backup tables in the db (check)

#### Batch 1

- Refined attempt
- Full DONs were pieced together from all relevant fields
- The documents were distilled into Markdown storage for the db
- Vectors were then taken from them for both nomic and embed-all (sometimes exceeding context window)
- Side quest: The Markdown docs were summarized with gpt-4o-mini
- Could make a Batch 2 with embeddings for these summaries
- Alternatively, these could be embedded inline as processing of the requests happens (though embedding is slow)


## Cosine Similarity in Vector Search

### What is Cosine Similarity?

Cosine similarity measures the cosine of the angle between two vectors in a multi-dimensional space. It's a measure of orientation rather than magnitude.

- Range: -1 to 1 (for normalized vectors, which is typical in text embeddings)
- 1: Vectors point in the same direction (most similar)
- 0: Vectors are orthogonal (unrelated)
- -1: Vectors point in opposite directions (most dissimilar)

### Cosine Distance

In pgvector, the `<=>` operator computes cosine distance, which is 1 - cosine similarity.

- Range: 0 to 2
- 0: Identical vectors (most similar)
- 1: Orthogonal vectors
- 2: Opposite vectors (most dissimilar)

### Interpreting Results

When you get results from similarity_search:

- Lower distance values indicate higher similarity.
- A distance of 0 would mean exact match (rarely happens with embeddings).
- Distances closer to 0 indicate high similarity.
- Distances around 1 suggest little to no similarity.
- Distances approaching 2 indicate opposite meanings (rare in practice).


## References

### YouTube

- [Reliable, fully local RAG agents with LLaMA3.2-3b - Langchain](https://www.youtube.com/watch?v=bq1Plo2RhYI)
- [Generate LLM Embeddings On Your Local Machine](https://www.youtube.com/watch?v=8L3tGcYc774&t=29s)
- [Don’t Embed Wrong! - Matt Williams](https://www.youtube.com/watch?v=76EIC_RaDNw)
- [Python RAG Tutorial (with Local LLMs): AI For Your PDFs – pixegami](https://www.youtube.com/watch?v=2TJxpyO3ei4)
- [AI for Good: Defeating Dengue with AI](https://www.youtube.com/watch?v=kPevp4f2CWw)


### Articles

- [Retrieval Augmented Generation (RAG) with pgvector vector database](https://medium.com/@yogi_r/retrieval-augmented-generation-rag-with-pgvector-vector-database-0d741e14d62f)


### Repos

- [Building a High-Performance RAG Solution with Pgvectorscale and Python](https://github.com/daveebbelaar/pgvectorscale-rag-solution/tree/setup)
- [https://github.com/ryogesh/llm-rag-pgvector](https://github.com/ryogesh/llm-rag-pgvector)
- [Swiss TPH OpenMalaria Wiki](https://github.com/SwissTPH/openmalaria/wiki)
- [technovangelist](https://github.com/technovangelist)
- [https://github.com/AlbertoFormaggio1/conversational_rag_web_interface](https://github.com/AlbertoFormaggio1/conversational_rag_web_interface)
- [https://github.com/nlmatics/nlm-ingestor](https://github.com/nlmatics/nlm-ingestor)
- [https://github.com/nlmatics/llmsherpa](https://github.com/nlmatics/llmsherpa)
- [https://github.com/segment-any-text/wtpsplit](https://github.com/segment-any-text/wtpsplit)
- [https://github.com/aws-samples/layout-aware-document-processing-and-retrieval-augmented-generation](https://github.com/aws-samples/layout-aware-document-processing-and-retrieval-augmented-generation)
- [https://github.com/aurelio-labs/semantic-chunkers](https://github.com/aurelio-labs/semantic-chunkers)


### Data

- [UNData](https://data.un.org/Data.aspx?d=WHO&f=MEASURE_CODE%3aWHS3_48)
- [WHO Malaria Factsheet](https://www.who.int/news-room/fact-sheets/detail/malaria)
- [Swiss TPH - Malaria](https://www.swisstph.ch/en/topics/malaria)

### Scientific Papers

- [Leveraging computational tools to combat malaria: assessment and development of new therapeutics](https://link.springer.com/article/10.1186/s13321-024-00842-z?fromPaywallRec=false)
- [Systematic review on the application of machine learning to quantitative structure–activity relationship modeling against Plasmodium falciparum](https://link.springer.com/article/10.1007/s11030-022-10380-1)
- [Predicting malaria outbreaks using earth observation measurements and spatiotemporal deep learning modelling: a South Asian case study from 2000 to 2017](https://www.thelancet.com/journals/lanplh/article/PIIS2542-5196(24)00082-2/fulltext)
- [New Study uses AI to predict malaria outbreaks in South Asia](https://www.ndorms.ox.ac.uk/news/new-study-uses-ai-to-predict-malaria-outbreaks-in-south-asia)

### Technical Articles

- [Load vector embeddings up to 67x faster with pgvector and Amazon Aurora](https://aws.amazon.com/blogs/database/load-vector-embeddings-up-to-67x-faster-with-pgvector-and-amazon-aurora/)
- [TF-IDF and BM25 for RAG— a complete guide](https://www.ai-bites.net/tf-idf-and-bm25-for-rag-a-complete-guide/)
- [Chunking Strategies for LLM Applications](https://www.pinecone.io/learn/chunking-strategies/)
- [Simplifying RAG with PostgreSQL and PGVector](https://medium.com/@levi_stringer/rag-with-pg-vector-with-sql-alchemy-d08d96bfa293)
- [Unleashing the power of vector embeddings with PostgreSQL](https://tembo.io/blog/pgvector-and-embedding-solutions-with-postgres)
- [PostgreSQL Extensions: Turning PostgreSQL Into a Vector Database With pgvector](https://www.timescale.com/learn/postgresql-extensions-pgvector)
- [Late Chunking in Long-Context Embedding Models](https://jina.ai/news/late-chunking-in-long-context-embedding-models/)
- [Chunk + Document Hybrid Retrieval with Long-Context Embeddings (Together.ai)](https://docs.llamaindex.ai/en/stable/examples/retrievers/multi_doc_together_hybrid/)
- [Retrieval Augmented Generation (RAG) for LLMs](https://www.promptingguide.ai/research/rag)
- [Build your RAG web application with Streamlit](https://medium.com/@alb.formaggio/build-your-rag-web-application-with-streamlit-7673120a9741)
- [Auto-Merging: RAG Retrieval Technique](https://dev.to/rutamstwt/auto-merging-rag-retrieval-technique-4d6m)
