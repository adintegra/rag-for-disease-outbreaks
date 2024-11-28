## MAS Master Thesis

**How can multi-modal data fusion inform and enhance the prediction and reporting of malaria outbreaks?**

- [MAS Master Thesis](#mas-master-thesis)
- [Introduction](#introduction)
- [Planning](#planning)
- [Code Environment](#code-environment)
- [Data Sources](#data-sources)
- [Data Acquisition](#data-acquisition)
  - [Approach 1: RAG](#approach-1-rag)
- [References](#references)
  - [YouTube](#youtube)
  - [Articles](#articles)
  - [Repos](#repos)
  - [Data](#data)
  - [Scientific Papers](#scientific-papers)
  - [Technical Articles](#technical-articles)


## Introduction

TODO:

## Planning

TODO: embed a project plan


## Code Environment

On MacOS you must install a native ARM build if you are running on Apple Silicon (Mn processors). Otherwise, Python will default to x86 builds which will run on Rosetta and ML will not run at all. See also [here](https://stackoverflow.com/questions/65415996/how-to-specify-the-architecture-or-platform-for-a-new-conda-environment-apple).

```
CONDA_SUBDIR=osx-arm64 conda create --name pg-vector-rag python=3.12 -c conda-forge

conda env remove --name pg-vector-rag

pip install -r requirements.txt
```

## Data Sources

Data source of interest will be found in the [./data](./data/) folder.

## Data Acquisition

See [./app/who-don-retriever/](./app/who-don-retriever/) for scripts to scrape the data.

### Approach 1: RAG




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
-


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
