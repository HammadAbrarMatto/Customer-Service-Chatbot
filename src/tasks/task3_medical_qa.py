"""
task3_medical_qa.py - INTERNSHIP TASK 3
Medical Q&A Chatbot using MedQuAD Dataset.
"""

import os
import glob
import csv
import xml.etree.ElementTree as ET

from langchain_community.vectorstores import FAISS
from langchain.schema import Document

from langchain_helper import embeddings, llm

MED_VECTORDB_PATH = "faiss_index_medical"
PREPARED_CSV = "dataset/medquad_prepared.csv"
MAX_DOCS = 50  # limit to avoid rate limits


def prepare_medquad_csv(medquad_folder: str, out_csv: str = PREPARED_CSV) -> int:
    rows = []
    for xml_file in glob.glob(os.path.join(medquad_folder, "**", "*.xml"), recursive=True):
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            for qapair in root.iter("QAPair"):
                q_node = qapair.find("Question")
                a_node = qapair.find("Answer")
                if q_node is not None and a_node is not None:
                    q = (q_node.text or "").strip()
                    a = (a_node.text or "").strip()
                    if q and a:
                        rows.append((q, a))
        except ET.ParseError:
            continue

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["question", "answer"])
        writer.writerows(rows)
    return len(rows)


def build_medical_db(csv_path: str = PREPARED_CSV) -> int:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found. Run prepare_medquad_csv() first.")

    docs = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            docs.append(Document(
                page_content=row["answer"],
                metadata={"question": row["question"]},
            ))
            if len(docs) >= MAX_DOCS:  # stop at 500
                break

    print(f"Building medical DB with {len(docs)} docs...")
    vectordb = FAISS.from_documents(docs, embeddings)
    vectordb.save_local(MED_VECTORDB_PATH)
    return len(docs)


SYMPTOMS = ["fever", "cough", "pain", "headache", "nausea", "fatigue", "rash",
            "swelling", "vomiting", "dizziness", "bleeding"]
DISEASES = ["diabetes", "cancer", "asthma", "hypertension", "covid", "flu",
            "arthritis", "anemia", "pneumonia", "migraine"]
TREATMENTS = ["surgery", "medication", "therapy", "antibiotics", "vaccine",
              "insulin", "chemotherapy", "physiotherapy"]


def recognise_entities(text: str) -> dict:
    lower = text.lower()
    return {
        "symptoms": [w for w in SYMPTOMS if w in lower],
        "diseases": [w for w in DISEASES if w in lower],
        "treatments": [w for w in TREATMENTS if w in lower],
    }


def answer_medical_question(question: str) -> dict:
    if not os.path.exists(MED_VECTORDB_PATH):
        raise FileNotFoundError("Medical database missing. Run build_medical_db() first.")

    vectordb = FAISS.load_local(
        MED_VECTORDB_PATH, embeddings, allow_dangerous_deserialization=True
    )
    results = vectordb.similarity_search(question, k=3)
    context = "\n\n".join(d.page_content for d in results)

    prompt = f"""You are a careful medical information assistant. Using ONLY the
context below, answer the user's question in simple language. If the context
does not contain the answer, say you do not have enough information and advise
seeing a doctor. Always remind the user this is general information, not a
diagnosis.

CONTEXT:
{context}

QUESTION: {question}
"""
    answer = llm.invoke(prompt)
    return {"answer": answer, "entities": recognise_entities(question)}


if __name__ == "__main__":
    print(answer_medical_question("What are the symptoms of diabetes?"))
