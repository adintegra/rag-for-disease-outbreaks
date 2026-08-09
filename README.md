# Enhancing Epidemiological Intelligence

**A RAG Approach to Disease Outbreak Monitoring**



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

`pyproject.toml` and `uv.lock` are the authoritative environment definition.

### Database Setup

This project uses the [pgvector](https://github.com/pgvector/pgvector) Postgres extension as a vector store. This allows the data to be stored alongside the embeddings and as such both can be accessed easily through any SQL querying utility.

If you do not wish to run this locally, a cloud-based service such as [Supabase](https://supabase.com/modules/vector) could also be used.

There is a [docker-compose.yml](./app/docker/docker-compose.yml) which sets up a local PGVector instance as a [Docker](https://www.docker.com/products/docker-desktop/) container. **Note:** Please create an empty subdirectory `pgvector_data` before bringing the container up for the first time. This will be mounted as a volume within the Docker container.

```sh
mkdir -p app/docker/pgvector_data
docker compose -f app/docker/docker-compose.yml up -d
```

For Neon or another PostgreSQL database, configure `DATABASE_URL` in the root `.env`, then apply the versioned schema migrations from the repository root:

```sh
uv run alembic upgrade head
```

The migration enables pgvector and creates persistent `ingest` staging and canonical `rag` schemas.

Run the complete idempotent pipeline manually with:

```sh
uv run python -m app.ingestion run
```

This acquires a PostgreSQL advisory lock and runs extraction, transformation, each profile in `INGESTION_CHUNK_PROFILES`, and incremental embeddings. For maintenance while the local embedding endpoint is unavailable, use `--skip-embeddings`; a later normal run will fill missing vectors.


### DB Schema

```mermaid
erDiagram
  INGEST_RUN ||--o{ RAW_RECORD : contains
  DOCUMENT ||--o{ CHUNK_DATASET : produces
  CHUNK_DATASET ||--o{ CHUNK : contains
  CHUNK ||--o{ EMBEDDING : has

  INGEST_RUN {
    UUID id PK
    TEXT source
    TEXT status
    JSONB metadata
  }
  RAW_RECORD {
    BIGINT id PK
    UUID run_id FK
    TEXT source_id
    JSONB payload
    TEXT payload_hash
  }
  DOCUMENT {
    BIGINT id PK
    TEXT source
    TEXT source_id
    TEXT content_hash
    TIMESTAMPTZ published_at
  }
  CHUNK_DATASET {
    BIGINT id PK
    BIGINT document_id FK
    TEXT profile_name
    TEXT configuration_hash
    TEXT document_hash
  }
  CHUNK {
    BIGINT id PK
    BIGINT chunk_dataset_id FK
    INTEGER chunk_index
    TEXT contents
    TEXT content_hash
  }
  EMBEDDING {
    BIGINT id PK
    BIGINT chunk_id FK
    TEXT model
    TEXT model_version
    HALFVEC embedding
  }
```

Alembic migrations in `app/db/migrations/` are the schema source of truth.

### LangChain

LangChain provides the OpenAI-compatible embedding and chat adapters. SQLAlchemy and `pgvector` handle persistence and cosine retrieval directly, while LangSmith tracing is optional.

### Data

A list of interesting data sources pertaining to malaria and other tropical diseases can be found in the subfolders under [./data-collection](./data-collection/). The code in this repo currently uses data scraped from [WHO DONs](./data-collection/acquisition/WHO%20DONs/) to populate our RAG knowledgebase. It should be fairly straightforward to adapt it to other sources.

#### Acquisition & Pre-Processing

`app.ingestion.source.who.WhoDonClient` retrieves paginated Disease Outbreak News directly from WHO with bounded retries and request timeouts. Raw source observations are retained in `ingest.raw_record`; HTML sections are normalized to Markdown during the transformation stage.

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

Retrieval supports `--limit`, repeatable `--section`, `--source`, `--published-after`, and `--published-before` filters. Profiles are defined in `app/config/retrieval_profiles.toml`.

With a chat/instruct model configured as `LLM_MODEL` on the local OpenAI-compatible endpoint, generate an evidence-grounded answer:

```sh
uv run python -m app.generation \
  "What Ebola outbreaks were reported in Uganda?"
```

The Flask application uses the same retrieval and generation path. Run it with `uv run flask --app app.app run`.

For a host cron schedule, use absolute paths and ensure LM Studio is running. Example daily invocation at 04:17:

```cron
17 4 * * * cd /absolute/path/to/mas-master-thesis && /absolute/path/to/uv run python -m app.ingestion run >> /absolute/path/to/logs/ingestion.log 2>&1
```

The command exits non-zero on failure, prevents overlapping complete runs, and records final status plus stage metrics in `ingest.run`.


### Local Models

Embedding and generation use LangChain's OpenAI-compatible adapters. A local server such as LM Studio should expose `/v1/embeddings` and `/v1/chat/completions`; configure `EMBEDDING_BASE_URL`, `EMBEDDING_MODEL`, `LLM_BASE_URL`, and `LLM_MODEL` in the root `.env`. The current tested setup uses `text-embedding-all-minilm-l6-v2-embedding` for 384-dimensional embeddings and `qwen2.5-7b-instruct` for cited answers.

### UI

Run the Flask application from the repository root:

```sh
uv run flask --app app.app run
```

The chat interface is available at http://127.0.0.1:5000 and uses the same profile-aware retrieval and evidence-grounded generation modules as the CLIs.

![RAG chat UI](./app/static/rag-chat-ui.png)


## Evaluation

### Sample Questions

- In which countries is Malaria most prevalent?
- Which diseases are prevalent in Kenya?
- Which were the largest disease outbreaks in the last 20 years?
- Where were outbreaks with the most severe impacts, e.g. deaths?



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
