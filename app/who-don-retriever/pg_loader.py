from time import sleep
import pandas as pd
import json
import re
import requests
from bs4 import BeautifulSoup
import warnings
from bs4 import MarkupResemblesLocatorWarning


def remove_tags(string):
  """
  Remove HTML tags and specific HTML entities from a given string. It also removes single and double quotes, and other
  specific HTML entities.
  Args:
    string (str): The input string containing HTML tags and entities.
  Returns:
    str: The cleaned string with HTML tags and entities removed.
  """

  result = re.sub(r"<.*?>", "", string)
  result = re.sub(r"[\n\r]+", "", result)
  result = re.sub(r"&ndash;", "-", result)
  result = re.sub(r"&mdash;", "–", result)
  result = re.sub(r"&egrave;", "è", result)
  result = re.sub(r"&eacute;", "é", result)
  result = re.sub(r"&agrave;", "à", result)
  result = re.sub(r"&aacute;", "á", result)
  result = re.sub(r"&itilde;", "î", result)
  # result = re.sub(r'&ldquo;','',result)
  # result = re.sub(r'&rdquo;','',result)
  result = re.sub(r"&lsquo;", "", result)
  result = re.sub(r"&rsquo;", "", result)
  result = re.sub(r"&[a-z]{4,6};", " ", result)
  result = re.sub(r";", "", result)
  result = re.sub(r'"', "", result)

  return result.strip()


def remove_tags_bs(string):
  """
  Remove HTML tags and specific HTML entities from a given string. It also removes single and double quotes, and other
  specific HTML entities. Uses BeatifulSoup to parse the HTML.
  Args:
    string (str): The input string containing HTML tags and entities.
  Returns:
    str: The cleaned string with HTML tags and entities removed.
  """
  warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

  soup = BeautifulSoup(string, "html.parser")
  soup = soup.get_text(strip=True)
  # soup = soup.lower()

  soup = re.sub(r"[\n\r]+", " ", soup)
  soup = re.sub(r";", "", soup)
  soup = re.sub(r'"', "", soup)

  return soup


def load_json():
  """
  Load data from a JSON file, normalize it into a pandas DataFrame, clean the data, and save it to a CSV file.
  Args:
    None
  Returns:
    None
  """

  with open("dons.json") as json_file:
    data = json.load(json_file)

    # df = pd.read_json(data)
    # df = pd.json_normalize(data, "value")
    df = pd.json_normalize(data)

    # print(df.head())
    # print(df.columns)

    #  Drop columns
    df.drop(
      columns=[
        "EmergencyEvent.Id",
        "EmergencyEvent.LastModified",
        "EmergencyEvent.PublicationDate",
        "EmergencyEvent.DateCreated",
        "EmergencyEvent.IncludeInSitemap",
        "EmergencyEvent.SystemSourceKey",
        "EmergencyEvent.UrlName",
        "EmergencyEvent.Title",
        "EmergencyEvent.ItemDefaultUrl",
        "EmergencyEvent.healthtopics",
        "EmergencyEvent.EventId",
        "EmergencyEvent.healthtopictypes",
        "EmergencyEvent.Provider",
        "EmergencyEvent",
      ],
      inplace=True,
    )

    # Clean up the data
    df["Summary"] = df["Summary"].fillna("").apply(str)
    df["Title"] = df["Title"].fillna("").apply(str)
    df["TitleSuffix"] = df["TitleSuffix"].fillna("").apply(str)
    df["Epidemiology"] = df["Epidemiology"].fillna("").apply(str)
    df["Assessment"] = df["Assessment"].fillna("").apply(str)
    df["Overview"] = df["Overview"].fillna("").apply(str)

    df["Summary"] = df["Summary"].apply(lambda cw: remove_tags_bs(cw))
    df["Title"] = df["Title"].apply(lambda cw: remove_tags_bs(cw))
    df["TitleSuffix"] = df["TitleSuffix"].apply(lambda cw: remove_tags_bs(cw))
    df["Epidemiology"] = df["Epidemiology"].apply(lambda cw: remove_tags_bs(cw))
    df["Assessment"] = df["Assessment"].apply(lambda cw: remove_tags_bs(cw))
    df["Overview"] = df["Overview"].apply(lambda cw: remove_tags_bs(cw))

    df["Url"] = df["UrlName"].apply(
      lambda x: f"https://www.who.int/emergencies/disease-outbreak-news/item/{x}"
    )

    # print(df.head())
    df.to_csv("dons.csv", sep=";", index=False)


def retrieve_dons():
  """This is a function to retrieve the Disease Outbreak News from the WHO API. It takes about 5 minutes to run."""

  # url = 'https://www.who.int/api/emergencies/diseaseoutbreaknews?sf_provider=dynamicProvider372&sf_culture=en&$orderby=PublicationDateAndTime%20desc&$expand=EmergencyEvent&$select=Title,TitleSuffix,OverrideTitle,UseOverrideTitle,regionscountries,ItemDefaultUrl,FormattedDate,PublicationDateAndTime&%24format=json&%24top=20&%24skip={}&%24count=true'
  # url = "https://www.who.int/api/news/diseaseoutbreaknews?sf_culture=en&%24orderby=PublicationDate%20asc&%24select=Title,PublicationDate,Summary,Overview,DonId,Epidemiology,Assessment,TitleSuffix,UrlName&%24format=json&%24count=true&%24top="
  url = "https://www.who.int/api/emergencies/diseaseoutbreaknews?sf_provider=dynamicProvider372&sf_culture=en&$orderby=PublicationDateAndTime%20desc&$expand=EmergencyEvent&$select=Title,PublicationDate,Summary,Overview,DonId,Epidemiology,Assessment,TitleSuffix,UrlName&%24format=json&%24top=20&%24skip={}&%24count=true"

  all_data = []
  skip = 0
  total_records = 3124
  while skip < total_records:
    response = requests.get(url.format(skip))
    if response.status_code == 200:
      data = response.json()
      all_data.extend(data["value"])
      skip += 20
    else:
      print(f"Failed to retrieve data at skip={skip}")
      break
    sleep(1)

  with open("dons.json", "w") as json_file:
    json.dump(all_data, json_file, indent=4)


def main():
  # Uncomment for fresh data retrieval and pre-precessing
  # retrieve_dons()
  load_json()


## ------------------------------------------------------
if __name__ == "__main__":
  main()
