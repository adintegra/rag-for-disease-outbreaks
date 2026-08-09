from dotenv import load_dotenv
import os
from langchain_core.language_models import BaseLanguageModel
from langchain_postgres import PGVector
from ragas import SingleTurnSample
from ragas.metrics import AspectCritic
from ragas.llms import LangchainLLMWrapper

from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector

load_dotenv("../../.env")


test_data = {
  "user_input": "summarise given text\nThe company reported an 8% rise in Q3 2024, driven by strong performance in the Asian market. Sales in this region have significantly contributed to the overall growth. Analysts attribute this success to strategic marketing and product localization. The positive trend in the Asian market is expected to continue into the next quarter.",
  "response": "The company experienced an 8% increase in Q3 2024, largely due to effective marketing strategies and product adaptation, with expectations of continued growth in the coming quarter.",
}

llm = BaseLanguageModel(model=os.getenv("LLM"), base_url=os.getenv("OLLAMA_BASE_URL"))

evaluator_llm = LangchainLLMWrapper(llm)

test_data = {
  "user_input": "summarise given text\nThe company reported an 8% rise in Q3 2024, driven by strong performance in the Asian market. Sales in this region have significantly contributed to the overall growth. Analysts attribute this success to strategic marketing and product localization. The positive trend in the Asian market is expected to continue into the next quarter.",
  "response": "The company experienced an 8% increase in Q3 2024, largely due to effective marketing strategies and product adaptation, with expectations of continued growth in the coming quarter.",
}

metric = AspectCritic(
  name="summary_accuracy",
  llm=evaluator_llm,
  definition="Verify if the summary is accurate.",
)
test_data = SingleTurnSample(**test_data)
metric.single_turn_ascore(test_data)


def setup_store():
  """Set up the vector store."""
  embeddings = OllamaEmbeddings(
    model="nomic-embed-text", base_url=os.getenv("OLLAMA_BASE_URL")
  )
  vector_store = PGVector(
    embeddings=embeddings,
    collection_name="my_docs",
    connection=os.getenv("CONNECTION_STRING"),
  )
  return vector_store
