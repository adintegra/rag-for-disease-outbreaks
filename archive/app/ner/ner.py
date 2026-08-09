import pandas as pd
import spacy
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from tqdm import tqdm
import json


# Initialize geocoder
geolocator = Nominatim(user_agent="geo_locator", timeout=10)


def NER():
  # Load dataset
  file_path = "postgres.csv"  # Update this if the file is in a different location
  df = pd.read_csv(file_path, header=None, names=["document"])

  # Load SpaCy NER model
  nlp = spacy.load("en_core_web_sm")

  # Extract locations
  location_entities = set()
  for doc in nlp.pipe(
    df["document"].dropna(), batch_size=50, disable=["tagger", "parser"]
  ):
    for ent in doc.ents:
      if ent.label_ in [
        "GPE",
        "LOC",
      ]:  # GPE = geopolitical entity (countries, cities, etc.)
        location_entities.add(ent.text)

  # Convert to sorted list
  location_entities = sorted(location_entities)

  return location_entities


# Function to get geolocation data
def geocode_location(location):
  try:
    geo_info = geolocator.geocode(location, timeout=10)
    if geo_info:
      return {
        "name": location,
        "latitude": geo_info.latitude,
        "longitude": geo_info.longitude,
        "country": geo_info.address.split(",")[-1].strip(),
      }
  except GeocoderTimedOut:
    return None
  return None


def main():
  # Extract location entities
  e = NER()

  # Geocode all extracted locations
  geo_data = []
  for location in tqdm(e):
    geo_info = geocode_location(location)
    if geo_info:
      geo_data.append(geo_info)

  # Save output as JSON
  output_path = "locations.json"
  with open(output_path, "w") as f:
    json.dump({"locations": geo_data}, f, indent=4)

  print(f"Location data saved to {output_path}")


if __name__ == "__main__":
  main()
