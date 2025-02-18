from dotenv import load_dotenv
import os
import sqlalchemy as sa
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import pandas as pd
# from vector_store import Document, Embedding


def distplot():
  """Create a distribution plot for contents length."""
  engine = create_engine(os.getenv("CONNECTION_STRING"))

  # Define a metadata object
  metadata = sa.MetaData()

  # Reflect the documents table from the database
  documents_table = sa.Table("document", metadata, autoload_with=engine)

  # Open a connection to the database
  with engine.connect() as connection:
    # Create a SQL expression to retrieve the length of each content
    query_contents = sa.select(
      sa.func.array_length(
        sa.func.regexp_split_to_array(documents_table.c.contents, r"\s+"), 1
      ).label("content_length")
    ).where(documents_table.c.batch == 1)

    query_date = sa.select(documents_table.c.event_date).where(
      documents_table.c.batch == 1
    )

    # Set this
    # query = query_contents
    query = query_date

    # Execute the query and fetch results
    result = connection.execute(query).fetchall()

  # Convert the result to a Pandas DataFrame for easier plotting
  # df = pd.DataFrame(result, columns=["content_length"])
  df = pd.DataFrame(result, columns=["event_date"])

  # Plot the distribution of content lengths with 25 bins
  plt.figure(figsize=(10, 6))
  # plt.hist(df["content_length"], bins=25, color="skyblue", edgecolor="black")
  # plt.title("Distribution of Document Content Lengths")
  # plt.xlabel("Content Length (words), bin size = 25")
  plt.hist(df["event_date"], bins=25, color="skyblue", edgecolor="black")
  plt.title("Distribution of DON Events")
  plt.xlabel("DON Events, bin size = 25")
  plt.ylabel("Frequency")
  plt.grid(True)
  plt.show()


def main():
  load_dotenv("../../../.env")

  distplot()


## ------------------------------------------------------
if __name__ == "__main__":
  main()
