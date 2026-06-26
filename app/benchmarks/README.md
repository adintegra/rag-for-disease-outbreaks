## Benchmarking and Evaluation

### BM25 vs Embeddings

Run from `app/benchmarks/retrieval/bm25_v_embeddings/`:

```sh
python bm25_v_embeddings.py
```

### Chunking Strategies

Run in the following order from `app/benchmarks/retrieval/ragas_test/`:

```sh
chroma run --path ./vector_store
```

```sh
python PostgresDocumentLoader.py
python chunk_vectorize.py
python rag.py
python test_w_questions.py
```

Unique chunking experiments also live in `app/benchmarks/chunking/`.

### LangSmith / RAGAS

- LangSmith evaluation: `app/benchmarks/retrieval/langsmith_eval.py`
- RAGAS smoke test: `app/benchmarks/ragas_evaluation.py`
