# WHO DON Retriever

The code in this directory scrapes the WHO's DONs (Disease Outbreak News) [articles](https://www.who.int/emergencies/disease-outbreak-news) and saves them to a JSON file. `pg_loader.py` can also be used to convert the JSON file to CSV.

Currently (late October 2024), there are 3124 records spanning over 20 years of historical news articles.

```sh
python pg_loader.py
```
