"""
task1_knowledge_base.py
------------------------
INTERNSHIP TASK 1
"Implement a system for dynamically expanding the chatbot's knowledge base.
Create a mechanism to periodically update the vector database with new
information from specified sources."

What this does, in plain words:
- The chatbot starts with the FAQ CSV.
- This module lets you ADD new information later WITHOUT rebuilding everything
  from scratch. New text gets turned into vectors and merged into the existing
  FAISS database.
- New information can come from (a) a CSV file, or (b) a web page URL.
- A "last updated" log file records when each update happened, so you can also
  run this on a schedule (for example with a cron job) to update periodically.

Note: This needs the FAISS database to already exist. Build it once from the
main app or by running langchain_helper.py first.
"""

import os
import datetime

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_helper import embeddings, load_vector_db
from config import VECTORDB_PATH

# Simple text file that records when updates happened.
UPDATE_LOG = "knowledge_base_updates.log"


def _log_update(message: str) -> None:
    """Write one line with the current time to the update log."""
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(UPDATE_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{stamp}] {message}\n")


def add_documents_to_db(new_documents) -> int:
    """
    Merge a list of new documents into the existing FAISS database.

    Returns the number of documents added.
    """
    if not new_documents:
        return 0

    # Load the database we already have on disk.
    vectordb = load_vector_db()

    # Add the new documents (FAISS computes their vectors using our embeddings).
    vectordb.add_documents(new_documents)

    # Save the updated database back to disk.
    vectordb.save_local(VECTORDB_PATH)
    return len(new_documents)


def update_from_csv(csv_path: str, source_column: str = "prompt") -> int:
    """Add new question/answer rows from a CSV file into the knowledge base."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    loader = CSVLoader(file_path=csv_path, source_column=source_column, encoding="utf-8")
    docs = loader.load()

    added = add_documents_to_db(docs)
    _log_update(f"Added {added} rows from CSV: {csv_path}")
    return added


def update_from_url(url: str) -> int:
    """
    Add the text content of a web page into the knowledge base.

    Long pages are split into smaller chunks so the retriever works better.
    """
    loader = WebBaseLoader(url)
    raw_docs = loader.load()

    # Split big pages into ~1000 character chunks with small overlap so context
    # is not cut in half between chunks.
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = splitter.split_documents(raw_docs)

    added = add_documents_to_db(docs)
    _log_update(f"Added {added} chunks from URL: {url}")
    return added


# Manual test from inside the src folder:  python task1_knowledge_base.py
if __name__ == "__main__":
    # Example: add a web page. Replace with any source you want.
    count = update_from_url("https://en.wikipedia.org/wiki/Customer_service")
    print(f"Added {count} chunks. See {UPDATE_LOG} for the log.")
