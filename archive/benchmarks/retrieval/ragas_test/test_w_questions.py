from dotenv import load_dotenv
import os
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain import hub
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
import pandas as pd
from time import sleep

# Load environment variables
load_dotenv("../../../../.env")

# Init the embeddings model
embeddings = OllamaEmbeddings(
  model="nomic-embed-text", base_url=os.getenv("OLLAMA_BASE_URL")
)

llm = ChatOllama(model=os.getenv("LLM"), base_url=os.getenv("OLLAMA_BASE_URL"))

vector_store = Chroma(embedding_function=embeddings, persist_directory="./vector_store")

prompt = hub.pull("rlm/rag-prompt")
# prompt = hub.pull("rlm/rag-prompt-llama")

QUESTIONS = [
  "Where is Dengue prevalent in Africa?",
  "When was the first case of COVID-19 reported?",
  "What are the symptoms of Malaria?",
]


class State(TypedDict):
  question: str
  context: List[Document]
  answer: str


def retrieve(state: State):
  retrieved_docs = vector_store.similarity_search(state["question"])
  return {"context": retrieved_docs}


def generate(state: State):
  docs_content = "\n\n".join(doc.page_content for doc in state["context"])
  messages = prompt.invoke({"question": state["question"], "context": docs_content})
  response = llm.invoke(messages)
  return {"answer": response.content}


def main():
  graph_builder = StateGraph(State).add_sequence([retrieve, generate])
  graph_builder.add_edge(START, "retrieve")
  graph = graph_builder.compile()

  # Create empty DataFrame with required columns
  df = pd.DataFrame(columns=["question", "context", "answer", "metadata"])

  # During iteration, create a row per question
  results = []

  for question in QUESTIONS:
    result = graph.invoke({"question": question})

    print(f"Context: {result['context']}\n\n")
    print(f"Answer: {result['answer']}")
    print("\n\n")

    # Create a row with the current results
    row = {
      "question": question,
      "context": [doc.page_content for doc in result["context"]],
      "answer": result["answer"],
      "metadata": [doc.metadata for doc in result["context"]],
    }
    results.append(row)
    sleep(1)

    # Convert results to DataFrame after the loop
  df = pd.DataFrame(results)
  df.to_excel("results.xlsx", index=True)


if __name__ == "__main__":
  main()
