from dotenv import load_dotenv
import os
import sys

# Required to import from parent directory
sys.path.append("../../../")

from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
import numpy as np
from ragas import EvaluationDataset
from ragas import evaluate, RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness
from sqlalchemy import create_engine, text

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from db.vector_store import DocEmbeddingView


load_dotenv("../../../../.env")


def get_embeddings():
  """Retrieve all documents from the database and return as an array."""
  engine = create_engine(os.getenv("CONNECTION_STRING"))
  Session = sessionmaker(bind=engine)
  session = Session()

  result = None

  stmt = select(DocEmbeddingView.embedding).where(
    DocEmbeddingView.batch == 1, DocEmbeddingView.model == "nomic-embed-text"
  )

  try:
    result = session.execute(stmt).all()
  except Exception as e:
    print(e)

  res = np.empty((len(result), 768))

  for row in result:
    arr = row[0].to_numpy()
    np.append(res, arr[np.newaxis, :], axis=0)

  return res


# Create database connection
engine = create_engine(os.getenv("CONNECTION_STRING"))

sample_docs = []
sample_embeddings = get_embeddings()

# Read documents from database
with engine.connect() as connection:
  result = connection.execute(text("SELECT contents FROM document WHERE batch = 1"))
  sample_docs = [row[0] for row in result]


# Increase the timeout settings 5min
run_config = RunConfig(timeout=300.0)


class RAG:
  def __init__(self, model="gpt-4o"):
    self.llm = ChatOllama(model=os.getenv("LLM"), base_url=os.getenv("OLLAMA_BASE_URL"))
    self.embeddings = OllamaEmbeddings(
      model="nomic-embed-text", base_url=os.getenv("OLLAMA_BASE_URL")
    )
    self.doc_embeddings = None
    self.docs = None

  def load_documents(self, documents, e):
    """Load documents and compute their embeddings."""
    self.docs = documents
    self.doc_embeddings = e

  def get_most_relevant_docs(self, query):
    """Find the most relevant document for a given query."""
    # if not self.docs or not self.doc_embeddings:
    #   raise ValueError("Documents and their embeddings are not loaded.")

    query_embedding = self.embeddings.embed_query(query)
    similarities = [
      np.dot(query_embedding, doc_emb)
      / (np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb))
      for doc_emb in self.doc_embeddings
    ]
    most_relevant_doc_index = np.argmax(similarities)
    return [self.docs[most_relevant_doc_index]]

  def generate_answer(self, query, relevant_doc):
    """Generate an answer for a given query based on the most relevant document."""
    prompt = f"question: {query}\n\nDocuments: {relevant_doc}"
    messages = [
      (
        "system",
        "You are a helpful assistant that answers questions based on given documents only.",
      ),
      ("human", prompt),
    ]
    ai_msg = self.llm.invoke(messages)
    return ai_msg.content


# Initialize RAG instance
rag = RAG()

# Load documents
rag.load_documents(sample_docs, sample_embeddings)

# Query and retrieve the most relevant document
query = "Which countries in Africa is malaria most prevalent in?"
relevant_doc = rag.get_most_relevant_docs(query)

# Generate an answer
answer = rag.generate_answer(query, relevant_doc)

print(f"Query: {query}")
print(f"Relevant Document: {relevant_doc}")
print(f"Answer: {answer}")


# #  Step 2
# sample_queries = [
#   "Who introduced the theory of relativity?",
#   "Who was the first computer programmer?",
#   "What did Isaac Newton contribute to science?",
#   "Who won two Nobel Prizes for research on radioactivity?",
#   "What is the theory of evolution by natural selection?",
# ]

# expected_responses = [
#   "Albert Einstein proposed the theory of relativity, which transformed our understanding of time, space, and gravity.",
#   "Ada Lovelace is regarded as the first computer programmer for her work on Charles Babbage's early mechanical computer, the Analytical Engine.",
#   "Isaac Newton formulated the laws of motion and universal gravitation, laying the foundation for classical mechanics.",
#   "Marie Curie was a physicist and chemist who conducted pioneering research on radioactivity and won two Nobel Prizes.",
#   "Charles Darwin introduced the theory of evolution by natural selection in his book 'On the Origin of Species'.",
# ]

# dataset = []

# for query, reference in zip(sample_queries, expected_responses):
#   relevant_docs = rag.get_most_relevant_docs(query)
#   response = rag.generate_answer(query, relevant_docs)
#   dataset.append(
#     {
#       "user_input": query,
#       "retrieved_contexts": relevant_docs,
#       "response": response,
#       "reference": reference,
#     }
#   )

# evaluation_dataset = EvaluationDataset.from_list(dataset)


# # Step 3
# evaluator_llm = LangchainLLMWrapper(
#   ChatOllama(model=os.getenv("LLM"), base_url=os.getenv("OLLAMA_BASE_URL"))
# )


# result = evaluate(
#   dataset=evaluation_dataset,
#   metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness()],
#   llm=evaluator_llm,
#   run_config=run_config,
# )
# print(result)

# try:
#   result.upload()
# except Exception as e:
#   print(f"Failed to upload evaluation results: {e}")
