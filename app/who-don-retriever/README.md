# WHO DON Retriever

The code in this directory scrapes the WHO's DONs (Disease Outbreak News) [articles](https://www.who.int/emergencies/disease-outbreak-news) and saves them to a JSON file:

```sh
python pg_loader.py
```

Currently (early February 2025), there are 3136 records spanning over 20 years of historical news articles. As the base data for this state is already part of this repository (see [dons.json](./dons.json)), executing the above code will simply re-run the data pre-processing and conversion routines. Uncomment the pertinent code if you wish to re-retrieve the source data itself.
