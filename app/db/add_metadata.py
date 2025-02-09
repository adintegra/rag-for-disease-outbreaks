from dotenv import load_dotenv
import os
import logging
import json
from tqdm import tqdm
from datetime import datetime
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError


BATCH = 1


def update_event_dates():
  # Logging
  if os.getenv("LOGS"):
    logging.basicConfig(
      level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

  # Read the JSON file
  try:
    with open("../who-don-retriever/dons.json", "r") as file:
      dons_data = json.load(file)
  except FileNotFoundError:
    print("Error: dons.json file not found")
    return
  except json.JSONDecodeError:
    print("Error: Invalid JSON format in dons.json")
    return

  try:
    # Create engine and session
    engine = create_engine(os.getenv("CONNECTION_STRING"))
    Session = sessionmaker(bind=engine)
    session = Session()

    print("Number of DONs: ", len(dons_data))

    with tqdm(total=len(dons_data), desc="Updating metadata") as pbar:
      # Process each DON object
      for don in dons_data:
        # Extract the publication date and title
        pub_date = don.get("PublicationDateAndTime")
        url_match = (
          "https://www.who.int/emergencies/disease-outbreak-news/item/"
          + don.get("UrlName")
        )
        # pub_match = don.get("PublicationDate")
        # title_match = don.get("Title")

        if pub_date:
          try:
            # Convert the date string to datetime object
            event_date = datetime.strptime(pub_date, "%Y-%m-%dT%H:%M:%SZ")
            # published_at = datetime.strptime(pub_match, "%Y-%m-%dT%H:%M:%SZ")

            # Update the document table
            stmt = text(
              "UPDATE document SET event_date = :event_date WHERE batch = :batch and url = :url"
            )
            session.execute(
              stmt,
              {
                "event_date": event_date,
                "batch": BATCH,
                "url": url_match,
              },
            )

          except ValueError:
            print(f"Error: Invalid date format for DON with URL: {url_match}")
            continue
        pbar.update(1)

      # Commit the changes
      session.commit()

  except SQLAlchemyError as e:
    print(f"Database error: {e}")
    session.rollback()

  finally:
    # Close session
    session.close()


def extract_unique_regions():
  # Read the JSON file
  try:
    with open("../who-don-retriever/dons.json", "r") as file:
      dons_data = json.load(file)
  except FileNotFoundError:
    print("Error: dons.json file not found")
    return
  except json.JSONDecodeError:
    print("Error: Invalid JSON format in dons.json")
    return

  # Extract unique region IDs
  unique_regions = set()
  for don in dons_data:
    if "regionscountries" in don:
      unique_regions.update(don["regionscountries"])

  # print("Unique region IDs:")
  # for region in sorted(unique_regions):
  #   print(region)

  return unique_regions


def insert_country_codes():
  unique_regions = extract_unique_regions()

  try:
    engine = create_engine(os.getenv("CONNECTION_STRING"))
    Session = sessionmaker(bind=engine)
    session = Session()

    for region in unique_regions:
      stmt = text("INSERT INTO country_lookup (country_code) VALUES (:code)")
      session.execute(stmt, {"code": region})

    session.commit()

  except SQLAlchemyError as e:
    print(f"Database error: {e}")
    session.rollback()

  finally:
    session.close()


def insert_document_countries():
  try:
    with open("../who-don-retriever/dons.json", "r") as file:
      dons_data = json.load(file)
  except FileNotFoundError:
    print("Error: dons.json file not found")
    return
  except json.JSONDecodeError:
    print("Error: Invalid JSON format in dons.json")
    return

  try:
    engine = create_engine(os.getenv("CONNECTION_STRING"))
    Session = sessionmaker(bind=engine)
    session = Session()

    with tqdm(total=len(dons_data), desc="Mapping countries") as pbar:
      for don in dons_data:
        if "regionscountries" not in don:
          continue

        url = (
          "https://www.who.int/emergencies/disease-outbreak-news/item/" + don["UrlName"]
        )

        # Get document id
        doc_stmt = text("SELECT id FROM document WHERE url = :url AND batch = :batch")
        result = session.execute(doc_stmt, {"url": url, "batch": BATCH}).first()

        if not result:
          continue

        document_id = result[0]

        # Insert country mappings
        for country_code in don["regionscountries"]:
          # Get country id
          country_stmt = text(
            "SELECT id FROM country_lookup WHERE country_code = :code"
          )
          country_result = session.execute(country_stmt, {"code": country_code}).first()

          if country_result:
            country_id = country_result[0]
            # Insert mapping
            mapping_stmt = text(
              "INSERT INTO document_country (document_id, country_id) "
              "VALUES (:doc_id, :country_id)"
            )
            session.execute(
              mapping_stmt, {"doc_id": document_id, "country_id": country_id}
            )

        pbar.update(1)

    session.commit()

  except SQLAlchemyError as e:
    print(f"Database error: {e}")
    session.rollback()
  finally:
    session.close()


def update_country_names():
  try:
    # Read the JSON file
    with open("../who-don-retriever/countries.json", "r") as file:
      countries_data = json.load(file)
  except FileNotFoundError:
    print("Error: countries.json file not found")
    return
  except json.JSONDecodeError:
    print("Error: Invalid JSON format in countries.json")
    return

  # Extract data from the 'value' array
  countries_data = countries_data.get("value", [])
  if not countries_data:
    print("Error: No country data found in 'value' array")
    return

  print("Number of countries: ", len(countries_data))

  try:
    engine = create_engine(os.getenv("CONNECTION_STRING"))
    Session = sessionmaker(bind=engine)
    session = Session()

    with tqdm(total=len(countries_data), desc="Updating country names") as pbar:
      for country in countries_data:
        country_code = country.get("Id")
        country_name = country.get("Title")

        if country_code and country_name:
          stmt = text(
            "UPDATE country_lookup SET country_name = :name WHERE country_code = :code"
          )
          session.execute(stmt, {"name": country_name, "code": country_code})

        pbar.update(1)

    session.commit()

  except SQLAlchemyError as e:
    print(f"Database error: {e}")
    session.rollback()
  finally:
    session.close()


def main():
  # Load environment variables
  load_dotenv("../../.env")

  # The following functions are used to seed the database with metadata
  # Generally, they only need to be run once
  # insert_country_codes()
  # update_event_dates()
  # insert_document_countries()
  update_country_names()


if __name__ == "__main__":
  main()
