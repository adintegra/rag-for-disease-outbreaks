from dotenv import load_dotenv
import os
import gradio as gr
from openai import OpenAI
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from db.vector_store import DocEmbeddingView


load_dotenv("../.env")


# Initialize the embedding models and vector store
client = OpenAI(api_key="***", base_url=os.getenv("OLLAMA_BASE_URL") + "/v1/")
start_message = [{"role": "system", "content": "You are an intelligent assistant."}]
engine = create_engine(os.getenv("CONNECTION_STRING") + "/postgres")


def embed_query(query, model="all-minilm"):
  """Embed a query using the specified model."""
  embeddings = OllamaEmbeddings(model=model, base_url=os.getenv("OLLAMA_BASE_URL"))
  return embeddings.embed_query(query)


def similarity_search(query, model="all-minilm"):
  """Search for documents directly using pgvector."""
  query_vector = embed_query(query, model)

  # See https://github.com/pgvector/pgvector-python?tab=readme-ov-file#sqlalchemy
  # l2_distance also works
  stmt = (
    select(
      DocEmbeddingView.document_id,
      DocEmbeddingView.contents,
      DocEmbeddingView.url,
      DocEmbeddingView.embedding.cosine_distance(query_vector).label("l2_distance"),
    )
    .filter(DocEmbeddingView.model == model)
    .filter(DocEmbeddingView.batch == 0)
    .order_by(DocEmbeddingView.embedding.cosine_distance(query_vector))
    .limit(5)
  )

  Session = sessionmaker(bind=engine)
  session = Session()

  retrieved_docs = session.execute(stmt).all()

  if len(retrieved_docs) == 0:
    raise ValueError("No documents found.")

  # else:
  #   print(f"Number of documents retrieved: {len(retrieved_docs)}")
  #   # for doc in retrieved_docs:
  #   #   print(f"* {doc.id}")
  #   #   print(f"URL: {doc.url}")
  #   #   print(f"Similarity: {doc.l2_distance}")

  return retrieved_docs


def post_prompt(docs):
  """."""
  formatted_response = "<br /><br />- - - - - - - - - - - -<br />Here are the retrieved documents most relevant to the query:<br /><br />"

  for doc in docs:
    formatted_response += f"<a href='{doc.url}' target='_blank'>Document</a>"
    formatted_response += f"<p>Cosine Similarity: {doc.l2_distance}</p>"
    formatted_response += f"<p>Content: {doc.contents[:350]}...</p>"

  return formatted_response


def tuples_to_messages(history_tuples):
  history_messages = []
  for message_tuple in history_tuples:
    if message_tuple[0]:
      history_messages.append({"role": "user", "content": message_tuple[0]})
    if message_tuple[1]:
      history_messages.append({"role": "assistant", "content": message_tuple[1]})
  print(history_messages)
  return history_messages


def format_conversation(history: list, new_message: str):
  conversation = []
  for message in history:
    if isinstance(message["content"], str):
      conversation.append({"role": message["role"], "content": message["content"]})
  print(conversation)
  return conversation


def rag_client(message, history, em, llm):
  """TBC"""
  history = history or start_message

  context = similarity_search(message, em)

  # Create a prompt that combines the query and retrieved documents
  prompt_text = f"""Answer the following question based on the provided context:
Question: {message}
Context:
"""
  for doc in context:
    prompt_text += f"{doc.contents}\nNotification URL: {doc.url}\nCosine Similarity: {doc.l2_distance}\n\n"

  prompt_text += "\nAnswer:"

  if len(history) > 0 and isinstance(history[0], (list, tuple)):
    history = tuples_to_messages(history)
  conversation = format_conversation(history, message)

  print(prompt_text)

  stream = client.chat.completions.create(
    messages=[
      {
        "role": "user",
        "content": prompt_text,
      }
    ],
    model=llm,
    stream=True,
  )

  response = ""
  for chunk in stream:
    if chunk.choices[0].delta.content is not None:
      response += chunk.choices[0].delta.content
      #   yield {"role": "assistant", "content": chunk.choices[0].delta.content or ""}
      yield response

  # return post_prompt(context)


with gr.Blocks(fill_height=True) as demo:
  # Sidebar for chat history
  # with gr.Column(scale=1, min_width=150):
  with gr.Sidebar(label="Chat History"):
    with gr.Row(height="20vh"):
      em_dropdown = gr.Dropdown(
        choices=[
          ("all-MiniLM-L6-v2", "all-minilm"),
          ("Nomic v1.5", "nomic-embed-text"),
        ],
        label="Embedding Model",
        interactive=True,
      )
      llm_dropdown = gr.Dropdown(
        choices=[("Llama 3.2 3B", "llama3.2_16kctx"), ("Phi-4 14B", "phi4_16kctx")],
        label="LLM",
        interactive=True,
      )
    chat_history = gr.Chatbot(
      label="Chat History", elem_id="chat-history", min_height="70vh", type="messages"
    )

  with gr.Column(scale=3):
    gr.Markdown(
      "# Disease Outbreak Chatbot\n\nAsk questions about disease outbreaks. Use the dropdowns below to select the models you wish to use.",
      min_height="10vh",
      line_breaks=True,
    )

    chatbot = gr.ChatInterface(
      fn=rag_client,
      additional_inputs=[em_dropdown, llm_dropdown],
      type="messages",
      description="---",
      chatbot=gr.Chatbot(height="50vh", type="messages"),
      save_history=True,
      analytics_enabled=False,
      examples=[
        [
          "What are the most common diseases reported in the dataset?",
          "all-minilm",
          "llama3.2_16kctx",
        ],
        [
          "Which regions have reported the highest number of Ebola outbreaks?",
          "all-minilm",
          "llama3.2_16kctx",
        ],
        [
          "How effective have quarantine measures been in containing outbreaks?",
          "all-minilm",
          "llama3.2_16kctx",
        ],
      ],
    )


if __name__ == "__main__":
  demo.launch()


# ["What are the common causes of recurring outbreaks?", "", "Nomic", "llama3.2"],
# [
#   "What financial burdens do outbreaks place on affected countries?",
#   "",
#   "Nomic",
#   "llama3.2",
# ],
# [
#   "How do vaccines contribute to reducing outbreak severity?",
#   "",
#   "Nomic",
#   "llama3.2",
# ],

# cache_examples=True,
# textbox=gr.Textbox(
#   placeholder="Ask me a yes or no question", container=False, scale=7
# ),
