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
    query = sa.select(
      sa.func.array_length(
        sa.func.regexp_split_to_array(documents_table.c.contents, r"\s+"), 1
      ).label("content_length"),
      documents_table.c.event_date,
    ).where(documents_table.c.batch == 1)

    # Execute the query and fetch results
    result = connection.execute(query).fetchall()

  # Convert the result to a Pandas DataFrame for easier plotting
  # df = pd.DataFrame(result, columns=["content_length"])
  df = pd.DataFrame(result, columns=["content_length", "event_date"])

  # Plot the distribution of content lengths with 25 bins
  # plt.figure(figsize=(10, 6))
  # # plt.hist(df["content_length"], bins=25, color="skyblue", edgecolor="black")
  # # plt.title("Distribution of Document Content Lengths")
  # # plt.xlabel("Content Length (words), bin size = 25")
  # plt.hist(df["event_date"], bins=25, color="skyblue", edgecolor="black")
  # plt.title("Distribution of DON Events")
  # plt.xlabel("DON Events, bin size = 25")
  # plt.ylabel("Frequency")
  # plt.grid(True)
  # plt.show()

  plt.figure(figsize=(12, 6))
  plt.hist2d(df["event_date"], df["content_length"], bins=(50, 50))
  plt.colorbar(label="Count")
  plt.title("Document Length vs Time")
  plt.xlabel("Date")
  plt.ylabel("Content Length (words)")

  # Format x-axis to show dates properly
  plt.gcf().autofmt_xdate()  # Rotate and align the tick labels
  plt.gca().xaxis.set_major_formatter(plt.FixedFormatter("%Y-%m-%d"))

  plt.show()


def timeseries_plot():
  """Create a plot of average document lengths over time."""
  engine = create_engine(os.getenv("CONNECTION_STRING"))
  metadata = sa.MetaData()
  documents_table = sa.Table("document", metadata, autoload_with=engine)

  with engine.connect() as connection:
    query = sa.select(
      sa.func.array_length(
        sa.func.regexp_split_to_array(documents_table.c.contents, r"\s+"), 1
      ).label("content_length"),
      documents_table.c.event_date,
    ).where(documents_table.c.batch == 1)
    result = connection.execute(query).fetchall()

  df = pd.DataFrame(result, columns=["content_length", "event_date"])

  # Resample data by 6-month periods and calculate mean
  df.set_index("event_date", inplace=True)
  df_resampled = df.resample("6M").median()
  # df_resampled2 = df.resample("6M").median()
  plt.figure(figsize=(12, 6))
  plt.plot(df_resampled.index, df_resampled["content_length"], "b-", marker="o")
  # plt.plot(df_resampled2.index, df_resampled["content_length"], "r-", marker="o")
  plt.title("Average Document Length Over Time (6-month intervals)")
  plt.xlabel("Date")
  plt.ylabel("Average Content Length (words)")
  plt.grid(True)
  plt.gcf().autofmt_xdate()
  plt.show()


def main():
  load_dotenv("../../../.env")

  # distplot()
  timeseries_plot()


## ------------------------------------------------------
if __name__ == "__main__":
  main()
