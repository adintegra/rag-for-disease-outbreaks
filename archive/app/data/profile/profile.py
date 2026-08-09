from dotenv import load_dotenv
import os
import pandas as pd
from ydata_profiling import ProfileReport
from sqlalchemy import create_engine, text


def profile():
  # df = pd.read_csv("../data/dons_md.csv", sep=";")
  # df = pd.read_json("../who-don-retriever/dons.json")

  engine = create_engine(os.getenv("CONNECTION_STRING"))

  # Execute query and load into DataFrame
  query = "SELECT * FROM document WHERE batch = 1"
  df = pd.read_sql_query(text(query), engine.connect())

  # Generate profiling report
  profile = ProfileReport(df, title="Profiling Report")
  profile.to_file("report_db.html")


def main():
  load_dotenv("../../../.env")
  profile()


## ------------------------------------------------------
if __name__ == "__main__":
  main()
