from dotenv import load_dotenv
import os
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain import hub
from langchain_core.documents import Document
from typing_extensions import List, TypedDict
from langgraph.graph import START, StateGraph
from langsmith import traceable

APP_PATH = "/Users/mark/Documents/Development/_Repositories/Adintegra/mas-master-thesis"

# Load environment variables
# load_dotenv("../../../../.env")
load_dotenv(APP_PATH + "/.env")

# Init the embeddings model
embeddings = OllamaEmbeddings(
  model="nomic-embed-text", base_url=os.getenv("OLLAMA_BASE_URL")
)

llm = ChatOllama(model=os.getenv("LLM"), base_url=os.getenv("OLLAMA_BASE_URL"))
# llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.5)

# vector_store = Chroma(embedding_function=embeddings, persist_directory="./vector_store")
vector_store = Chroma(
  embedding_function=embeddings,
  persist_directory=APP_PATH + "/app/benchmarks/retrieval/ragas_test/vector_store",
)

prompt = hub.pull("rlm/rag-prompt")
# prompt = hub.pull("rlm/rag-prompt-llama")


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


# Add decorator so this function is traced in LangSmith
@traceable()
def ask(question: str) -> dict:
  graph_builder = StateGraph(State).add_sequence([retrieve, generate])
  graph_builder.add_edge(START, "retrieve")
  graph = graph_builder.compile()

  # result = graph.invoke({"question": "Where is Dengue prevalent in Africa?"})
  result = graph.invoke({"question": question})

  # print(f"Context: {result['context']}\n\n")
  # print(f"Answer: {result['answer']}")
  # print(result)

  return {
    "documents": result["context"],
    "answer": result["answer"],
  }


if __name__ == "__main__":
  ask("Where is Dengue prevalent in Africa?")
