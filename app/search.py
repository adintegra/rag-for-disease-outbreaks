import logging
import sys

from core import config
from core.rag import generate_answer, similarity_search


def main():
  if len(sys.argv) < 2:
    logging.warning("Please provide a search query as a command line argument")
    sys.exit(1)

  query = " ".join(sys.argv[1:])
  docs = similarity_search(query)
  print(generate_answer(query, docs))


if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO)
  main()
