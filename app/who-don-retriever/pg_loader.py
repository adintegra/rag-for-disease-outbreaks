import pandas as pd
import json
import re
from langchain_community.llms import Ollama
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import declarative_base, sessionmaker


DB_URL = 'postgresql://postgres:password@127.0.0.1/postgres'

def remove_tags(string):
  result = re.sub(r'<.*?>','',string)
  result = re.sub(r'[\n\r]+','',result)
  result = re.sub(r'&ndash;','-',result)
  result = re.sub(r'&mdash;','–',result)
  result = re.sub(r'&egrave;','è',result)
  result = re.sub(r'&eacute;','é',result)
  result = re.sub(r'&agrave;','à',result)
  result = re.sub(r'&aacute;','á',result)
  result = re.sub(r'&itilde;','î',result)
  # result = re.sub(r'&ldquo;','',result)
  # result = re.sub(r'&rdquo;','',result)
  result = re.sub(r'&lsquo;','',result)
  result = re.sub(r'&rsquo;','',result)
  result = re.sub(r'&[a-z]{4,6};',' ',result)
  result = re.sub(r';','',result)
  result = re.sub(r'"','',result)

  return result.strip()

def load_json():

  with open('dons.json') as json_file:
    data = json.load(json_file)

    # df = pd.read_json(data)
    df = pd.json_normalize(data, "value")

    # Clean up the data
    df['Summary'] = df['Summary'].apply(lambda cw : remove_tags(cw))
    df['Epidemiology'] = df['Epidemiology'].apply(lambda cw : remove_tags(cw))
    df['Assessment'] = df['Assessment'].apply(lambda cw : remove_tags(cw))
    df['Overview'] = df['Overview'].apply(lambda cw : remove_tags(cw))

    # print(df.head())
    df.to_csv('dons.csv', sep=";", index=False)

def get_embeddings():
  ollm = Ollama(model="llama3.2")
  embeddings = ollm.embed(["hello", "world"])
  print(embeddings)

# API-based
# response = requests.get('https://api.example.com/data')
# data = response.json()
# print(data)

Base = declarative_base()

class Document(Base):
  __tablename__ = 'documents'
  id = Column(Integer, primary_key=True, autoincrement=True)
  title = Column(Text)
  summary = Column(Text)
  overview = Column(Text)
  publicationDate = Column(String)
  donId = Column(String)
  meta = Column(JSONB)

def create_table():
  engine = create_engine(DB_URL)
  Base.metadata.create_all(engine)


# create_table()
load_json()
