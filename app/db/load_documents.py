from dotenv import load_dotenv
import os
import logging
from datetime import datetime
import re
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from vector_store import Document
from langchain_ollama import OllamaLLM, ChatOllama
from langchain.prompts import PromptTemplate
import boto3
from botocore.exceptions import ClientError
import json

# Prepare data for insertion
def prepare_record(row):
  """Prepare a record for insertion into the database."""

  # Get rid of NaN values
  row = row.fillna("")

  title = row["Title"] if row["Title"] != "" else ""
  title_suffix = row["TitleSuffix"] if row["TitleSuffix"] != "" else ""
  epidemiology = row["Epidemiology"] if row["Epidemiology"] != "" else ""
  assessment = row["Assessment"] if row["Assessment"] != "" else ""
  overview = row["Overview"] if row["Overview"] != "" else ""
  summary = row["Summary"] if row["Summary"] != "" else ""

  # content = f"Title: {row['Title']}\nSummary: {row['Summary'] if pd.notna(row['Summary']) else row['Overview']}"

  content = ""
  content += f"\nTitle: {title}" if title != "" else ""
  content += f"\nSubtitle: {title_suffix}" if title_suffix != "" else ""
  content += f"\nSummary: {summary}" if summary != "" else ""
  content += f"\nEpidemiology: {epidemiology}" if epidemiology != "" else ""
  content += f"\nAssessment: {assessment}" if assessment != "" else ""
  content += f"\nOverview: {overview}" if overview != "" else ""
  content = re.sub(r"^[\n\r]+", "", content)

  published = datetime.strptime(
    row["PublicationDate"], "%Y-%m-%dT%H:%M:%SZ"
  ).isoformat()

  emergency_start = (
    datetime.strptime(
      row["EmergencyEvent.EmergencyEventStartDate"], "%Y-%m-%dT%H:%M:%SZ"
    ).isoformat()
    if row["EmergencyEvent.EmergencyEventStartDate"] != ""
    else None
  )

  return pd.Series(
    {
      "meta": {
        "title": title,
        "title_suffix": title_suffix,
        "epidemiology": epidemiology,
        "assessment": assessment,
        "overview": overview,
        "summary": summary,
        "published_at": published,
        "emergency_start": emergency_start,
        "don_id": row["UrlName"],
        "url": row["Url"],
      },
      "contents": content,
      "url": row["Url"],
      "published_at": published,
      "summary": "",
      "batch": 0,
    }
  )


def upsert(df: pd.DataFrame):
  """Upsert records into the database."""
  engine = create_engine(os.getenv("CONNECTION_STRING"))
  Session = sessionmaker(bind=engine)
  session = Session()

  for i, row in df.iterrows():
    record = Document(**row)
    session.add(record)

  session.commit()
  session.close()


def summarize_documents():
  """Summarize documents using LangChain with Ollama's llama3.2 model."""

  # Setup database connection
  engine = create_engine(os.getenv("CONNECTION_STRING"))
  Session = sessionmaker(bind=engine)
  session = Session()

  # Initialize Ollama LLM
  llm = OllamaLLM(
    model=os.getenv("LLM"),
    base_url=os.getenv("OLLAMA_BASE_URL"),
    num_ctx=12000,
    temperature=0.5,
  )
  # llm = ChatOllama(
  #   model=os.getenv("LLM"), base_url=os.getenv("OLLAMA_BASE_URL"), num_ctx=12000
  # )

  # Create prompt template
  prompt_template = PromptTemplate(
    input_variables=["text"],
    template="Summarize the following text in no more than 500 words. Don't repeat yourself. Just provide an answer: {text}",
  )

  try:
    # Get all documents without summaries
    documents = session.query(Document).filter(Document.summary == "").all()

    for doc in documents:
      try:
        # Generate summary
        prompt = prompt_template.format(text=doc.contents)
        summary = llm.predict(prompt).strip()
        doc.summary = summary
        session.add(doc)
      except Exception as e:
        logging.error(f"Error processing document {doc.id}: {str(e)}")
        continue

      session.commit()

  except Exception as e:
    logging.error(f"Database error: {str(e)}")
    session.rollback()
  finally:
    session.close()


def summarize_documents_aws():
  brt = boto3.client(service_name="bedrock-runtime")

  body = json.dumps(
    {
      "prompt": "Explain rainfall in a simple way.",
      "max_gen_len": 512,
      "temperature": 0.5,
      "top_p": 0.9,
    }
  )

  modelId = "us.meta.llama3-2-3b-instruct-v1:0"

  try:
    response = brt.invoke_model(body=body, modelId=modelId)
  except (ClientError, Exception) as e:
    print(f"ERROR: Can't invoke '{modelId}'. Reason: {e}")
    exit(1)

  # Decode the response body.
  model_response = json.loads(response["body"].read())
  response_text = model_response["generation"]

  print(response_text)


def main():
  load_dotenv("../../.env")

  # Logging
  if os.getenv("LOGS"):
    logging.basicConfig(
      level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

  # Read the CSV file and prepare records
  df = pd.read_csv("../data/dons.csv", sep=";", encoding="utf-8")
  records_df = df.apply(prepare_record, axis=1)

  upsert(records_df)
  # summarize_documents()


## ------------------------------------------------------
if __name__ == "__main__":
  main()
