import os, sys, pprint
import pandas as pd
import numpy as np
import ragas, datasets

# Libraries to customize ragas critic model.
from ragas.llms import LangchainLLMWrapper
from langchain_community.chat_models import ChatOllama

# Libraries to customize ragas embedding model.
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper


import eval_ragas as _eval_ragas

# Import the evaluation metrics.
from ragas.metrics import (
  context_recall,
  context_precision,
  faithfulness,
  answer_relevancy,
  answer_similarity,
  answer_correctness,
)

# Get the current working directory.
cwd = os.getcwd()
relative_path = "./blog_eval_answers.csv"
file_path = cwd + relative_path
# print(f"file_path: {file_path}")

# Read ground truth answers from a CSV file.
eval_df = pd.read_csv(file_path, header=0, skip_blank_lines=True)
display(eval_df.head())


##########################################
# Set the evaluation type.
EVALUATE_WHAT = "ANSWERS"
EVALUATE_WHAT = "CONTEXTS"
##########################################

# Set the columns to evaluate.
if EVALUATE_WHAT == "CONTEXTS":
  cols_to_evaluate = [
    "recursive_context_512_k_2",
    "html_context_512_k_2",
    "parent_context_1536_k1",
    "semantic_context_k_1",
    "semantic_context_k_2_summary",
  ]
  # cols_to_evaluate=\
  # ['parent_context_1536_k1', 'parent_context_1536_k1_text-embedding-3-small']
elif EVALUATE_WHAT == "ANSWERS":
  cols_to_evaluate = [
    "Custom_RAG_answer",
    "llama3_ollama_answer",
    "llama3_anyscale_answer",
    "llama3_octoai_answer",
    "llama3_groq_answer",
    "mixtral_8x7b_anyscale_answer",
  ]

# Set the metrics to evaluate.
if EVALUATE_WHAT == "ANSWERS":
  eval_metrics = [
    answer_relevancy,
    answer_similarity,
    answer_correctness,
    faithfulness,
  ]
  metrics = [
    "answer_relevancy",
    "answer_similarity",
    "answer_correctness",
    "faithfulness",
  ]
elif EVALUATE_WHAT == "CONTEXTS":
  eval_metrics = [
    context_recall,
    context_precision,
  ]
  metrics = ["context_recall", "context_precision"]

# Change the default llm-as-critic model to gpt-3.5-turbo.
# LLM_NAME = "gpt-3.5-turbo" #OpenAI
# ragas_llm = ragas.llms.llm_factory(model=LLM_NAME)

# Change the default the llm-as-critic model to local llama3.
LLM_NAME = "llama3"
ragas_llm = LangchainLLMWrapper(langchain_llm=ChatOllama(model=LLM_NAME))

# Change the default embeddings models to use model on HuggingFace.
EMB_NAME = "BAAI/bge-large-en-v1.5"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
lc_embed_model = HuggingFaceEmbeddings(
  model_name=EMB_NAME, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)
ragas_emb = LangchainEmbeddingsWrapper(embeddings=lc_embed_model)

# Change embeddings and critic models for each metric.
for metric in metrics:
  globals()[metric].llm = ragas_llm
  globals()[metric].embeddings = ragas_emb

# Execute the evaluation.
print(f"Evaluating {EVALUATE_WHAT} using {eval_df.shape[0]} eval questions:")
ragas_result, scores = _eval_ragas.evaluate_ragas_model(
  eval_df,
  eval_metrics,
  what_to_evaluate=EVALUATE_WHAT,
  cols_to_evaluate=cols_to_evaluate,
)
