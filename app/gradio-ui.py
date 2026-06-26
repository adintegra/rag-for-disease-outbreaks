import gradio as gr
from openai import OpenAI

from core import config
from core.rag import build_prompt, similarity_search

client = OpenAI(api_key="ollama", base_url=config.ollama_base_url + "/v1/")


def rag_client(message, history, em, llm):
  docs = similarity_search(message, model=em)
  prompt_text = build_prompt(message, docs)

  stream = client.chat.completions.create(
    messages=[{"role": "user", "content": prompt_text}],
    model=llm,
    stream=True,
  )

  response = ""
  for chunk in stream:
    if chunk.choices[0].delta.content is not None:
      response += chunk.choices[0].delta.content
      yield response


with gr.Blocks(fill_height=True) as demo:
  with gr.Sidebar(label="Chat History"):
    with gr.Row(height="20vh"):
      em_dropdown = gr.Dropdown(
        choices=[
          ("all-MiniLM-L6-v2", "all-minilm"),
          ("Nomic v1.5", "nomic-embed-text"),
        ],
        label="Embedding Model",
        value=config.embedding_model,
        interactive=True,
      )
      llm_dropdown = gr.Dropdown(
        choices=[("Llama 3.2 3B", "llama3.2_16kctx"), ("Phi-4 14B", "phi4_16kctx")],
        label="LLM",
        value=config.llm,
        interactive=True,
      )
    gr.Chatbot(
      label="Chat History", elem_id="chat-history", min_height="70vh", type="messages"
    )

  with gr.Column(scale=3):
    gr.Markdown(
      "# Disease Outbreak Chatbot\n\nAsk questions about disease outbreaks.",
      min_height="10vh",
      line_breaks=True,
    )

    gr.ChatInterface(
      fn=rag_client,
      additional_inputs=[em_dropdown, llm_dropdown],
      type="messages",
      chatbot=gr.Chatbot(height="50vh", type="messages"),
      save_history=False,
      analytics_enabled=False,
    )


if __name__ == "__main__":
  demo.launch()
