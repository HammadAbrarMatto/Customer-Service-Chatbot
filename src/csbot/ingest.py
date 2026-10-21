"""Turn raw sources (FAQ CSV, web pages, MedQuAD, arXiv) into documents."""

from __future__ import annotations

import csv
import glob
import io
import json
import os
import re
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from pathlib import Path

from .vectorstore import Document


def chunk_text(text: str, size: int = 900, overlap: int = 150) -> list[str]:
    """Paragraph/sentence-aware chunking with character overlap."""
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    if len(text) <= size:
        return [text] if text else []
    pieces = re.split(r"(?<=[.!?])\s+|\n\n", text)
    chunks, cur = [], ""
    for p in pieces:
        if len(cur) + len(p) + 1 <= size:
            cur = f"{cur} {p}".strip()
            continue
        if cur:
            chunks.append(cur)
            cur = cur[-overlap:] + " " + p if overlap else p
        else:
            cur = p
        while len(cur) > size:
            chunks.append(cur[:size])
            cur = cur[size - overlap:]
    if cur.strip():
        chunks.append(cur.strip())
    return chunks


def faq_documents(rows: Iterable[dict], source: str = "faq") -> list[Document]:
    """FAQ rows → one document per Q/A pair. The question *and* answer are embedded."""
    docs = []
    for r in rows:
        q = (r.get("prompt") or r.get("question") or "").strip()
        a = (r.get("response") or r.get("answer") or "").strip()
        if q and a:
            docs.append(Document(text=f"Q: {q}\nA: {a}", metadata={"question": q, "answer": a, "source": source}))
    return docs


def load_faq_csv(path: str | Path, source: str | None = None) -> list[Document]:
    with open(path, encoding="utf-8") as f:
        return faq_documents(csv.DictReader(f), source or Path(path).name)


def load_faq_bytes(data: bytes, name: str) -> list[Document]:
    return faq_documents(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))), name)


def load_url(url: str, timeout: int = 15) -> list[Document]:
    import requests
    from bs4 import BeautifulSoup

    resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0 (CS-Chatbot ingest)"})
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]):
        tag.decompose()
    title = soup.title.get_text(strip=True) if soup.title else url
    root = soup.find("article") or soup.find("main") or soup.body or soup
    text = "\n\n".join(el.get_text(" ", strip=True) for el in root.find_all(["h1", "h2", "h3", "p", "li"]))
    return [Document(text=c, metadata={"source": url, "title": title}) for c in chunk_text(text)]


# ── MedQuAD ──────────────────────────────────────────────────────────────────
def prepare_medquad_csv(medquad_folder: str | Path, out_csv: str | Path) -> int:
    """Flatten the MedQuAD XML collection (github.com/abachaa/MedQuAD) into question,answer CSV."""
    rows = []
    for xml_file in glob.glob(os.path.join(medquad_folder, "**", "*.xml"), recursive=True):
        try:
            root = ET.parse(xml_file).getroot()
        except ET.ParseError:
            continue
        focus = (root.findtext("Focus") or "").strip()
        for qa in root.iter("QAPair"):
            q, a = (qa.findtext("Question") or "").strip(), (qa.findtext("Answer") or "").strip()
            if q and a:
                rows.append((q, a, focus))
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["question", "answer", "focus"])
        w.writerows(rows)
    return len(rows)


def load_medquad(csv_path: str | Path, limit: int | None = None) -> list[Document]:
    docs = []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for i, chunk in enumerate(chunk_text(row["answer"], size=1200)):
                docs.append(Document(text=f"{row['question']}\n{chunk}",
                                     metadata={"question": row["question"], "source": "MedQuAD", "part": i}))
            if limit and len(docs) >= limit:
                return docs[:limit]
    return docs


# ── arXiv ────────────────────────────────────────────────────────────────────
def load_arxiv(json_path: str | Path, category_prefix: str = "cs.", limit: int = 5000) -> list[Document]:
    """Read the Kaggle `arxiv-metadata-oai-snapshot.json` (JSON lines) and keep one category."""
    docs = []
    with open(json_path, encoding="utf-8") as f:
        for line in f:
            if len(docs) >= limit:
                break
            try:
                p = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not any(c.startswith(category_prefix) for c in p.get("categories", "").split()):
                continue
            title = " ".join(p.get("title", "").split())
            abstract = " ".join(p.get("abstract", "").split())
            if title and abstract:
                docs.append(Document(text=f"{title}. {abstract}", metadata={
                    "title": title, "id": p.get("id", ""), "categories": p.get("categories", ""),
                    "authors": p.get("authors", "")[:200], "abstract": abstract,
                    "url": f"https://arxiv.org/abs/{p.get('id', '')}", "source": "arXiv"}))
    return docs
