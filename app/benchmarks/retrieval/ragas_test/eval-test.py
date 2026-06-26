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

# Import custom functions for evaluation.
sys.path.append("../../Evaluation")
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
relative_path = "/../../Evaluation/data/blog_eval_answers.csv"
file_path = cwd + relative_path
# print(f"file_path: {file_path}")

# Read ground truth answers from a CSV file.
eval_df = pd.read_csv(file_path, header=0, skip_blank_lines=True)
display(eval_df.head())
