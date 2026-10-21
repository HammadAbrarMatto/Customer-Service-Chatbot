"""Build / update knowledge-base indexes.

    python -m csbot.cli build-faq                         # data/faq.csv → indexes/faq
    python -m csbot.cli add-csv new_faqs.csv              # merge more FAQ rows
    python -m csbot.cli add-url https://example.com/help  # merge a web page
    python -m csbot.cli prepare-medquad path/to/MedQuAD   # XML → data/medquad_prepared.csv
    python -m csbot.cli build-medical --limit 5000
    python -m csbot.cli build-arxiv arxiv-metadata-oai-snapshot.json --category cs. --limit 5000
    python -m csbot.cli ask "Do you provide internships?"
"""

from __future__ import annotations

import argparse
import sys

from .config import settings


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="csbot", description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build-faq")
    p = sub.add_parser("add-csv")
    p.add_argument("path")
    p = sub.add_parser("add-url")
    p.add_argument("url")
    p = sub.add_parser("prepare-medquad")
    p.add_argument("folder")
    p = sub.add_parser("build-medical")
    p.add_argument("--limit", type=int, default=5000, help="max chunks (0 = all ~47k)")
    p = sub.add_parser("build-arxiv")
    p.add_argument("json_path")
    p.add_argument("--category", default="cs.")
    p.add_argument("--limit", type=int, default=5000)
    p = sub.add_parser("ask")
    p.add_argument("question")
    a = ap.parse_args(argv)

    from . import ingest
    from .knowledge import get_kb

    print(f"[embedder] {settings.embedding_backend}")
    if a.cmd == "build-faq":
        n = get_kb("faq").rebuild(ingest.load_faq_csv(settings.faq_csv))
    elif a.cmd == "add-csv":
        n = get_kb("faq").add(ingest.load_faq_csv(a.path), source=a.path)
    elif a.cmd == "add-url":
        n = get_kb("faq").add(ingest.load_url(a.url), source=a.url)
    elif a.cmd == "prepare-medquad":
        n = ingest.prepare_medquad_csv(a.folder, settings.medquad_csv)
    elif a.cmd == "build-medical":
        n = get_kb("medical").rebuild(ingest.load_medquad(settings.medquad_csv, limit=a.limit or None))
    elif a.cmd == "build-arxiv":
        n = get_kb("arxiv").rebuild(ingest.load_arxiv(a.json_path, a.category, a.limit))
    else:
        from .chat import respond

        analysis, tokens = respond(a.question, [], get_kb("faq"))
        print(f"[{analysis.language} · {analysis.sentiment.label}] ", end="")
        for t in tokens:
            print(t, end="", flush=True)
        print()
        return 0
    print(f"[done] {n} items")
    return 0


if __name__ == "__main__":
    sys.exit(main())
