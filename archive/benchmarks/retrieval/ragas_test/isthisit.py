from dotenv import load_dotenv
import os
import sys

# Required to import from parent directory
sys.path.append("../../../")

import nest_asyncio
import PostgresDocumentLoader
from langchain_community.document_loaders import DirectoryLoader
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ragas.llms import LangchainLLMWrapper
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from ragas.testset import TestsetGenerator
from ragas.dataset_schema import EvaluationDataset
from ragas.testset.synthesizers import (
  SingleHopSpecificQuerySynthesizer,
  MultiHopAbstractQuerySynthesizer,
)


# Apply nest_asyncio to avoid event loop issues
nest_asyncio.apply()


# Load OpenAI API key from environment variables or .env file
load_dotenv("../../../../.env")  # Ensure you have a .env file with OPENAI_API_KEY


# Step 1: Document loader
loader = PostgresDocumentLoader.PostgresDocumentLoader()
docs = loader.load()


# Split the documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

embeddings = OllamaEmbeddings(
  model="nomic-embed-text", base_url=os.getenv("OLLAMA_BASE_URL")
)

# Alternatively?
vector_store = Chroma(embedding_function=embeddings, persist_directory="./vector_store")
# splits = vector_store.get()

# Wrap the LLM with LangchainLLMWrapper using OpenAI GPT-4 model
# evaluator_llm = LangchainLLMWrapper(
#   ChatOllama(model=os.getenv("LLM"), base_url=os.getenv("OLLAMA_BASE_URL"))
# )

# Initialize Groq LLM
evaluator_llm = LangchainLLMWrapper(
  ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.7)
)


# Generate the test set with the loaded documents (generating 30 examples)
generator = TestsetGenerator(llm=evaluator_llm, embedding_model=embeddings)


# Assuming the function signature doesn't accept `docs`, pass splits as positional argument
# dataset = generator.generate_with_langchain_docs(splits, testset_size=30)
query_distribution = [
  (MultiHopAbstractQuerySynthesizer(llm=evaluator_llm), 0.5),
  (SingleHopSpecificQuerySynthesizer(llm=evaluator_llm), 0.5),
]


# Call the generate_with_langchain_docs with the custom query_distribution
dataset = generator.generate_with_langchain_docs(
  splits, testset_size=30, query_distribution=query_distribution
)


# Convert the generated dataset to a Pandas DataFrame
df = dataset.to_pandas()
print(df)


# Optionally, save the generated testset to a CSV file for further inspection
output_csv_path = "generated_testset.csv"
df.to_csv(output_csv_path, index=False)
print(f"Generated testset saved to {output_csv_path}")
