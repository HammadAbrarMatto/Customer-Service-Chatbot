"""Task 3 — Medical Q&A over MedQuAD with lightweight entity recognition."""

from __future__ import annotations

import re

from .. import llm
from ..knowledge import KnowledgeBase

ENTITY_LEXICON = {
    "symptoms": ["fever", "cough", "pain", "headache", "nausea", "fatigue", "tiredness", "rash", "swelling",
                 "vomiting", "dizziness", "bleeding", "shortness of breath", "chest pain", "itching", "numbness",
                 "weight loss", "diarrhea", "constipation", "seizure", "blurred vision", "sore throat"],
    "diseases": ["diabetes", "cancer", "leukemia", "lymphoma", "asthma", "hypertension", "high blood pressure",
                 "covid", "influenza", "flu", "arthritis", "anemia", "pneumonia", "migraine", "alzheimer",
                 "parkinson", "stroke", "glaucoma", "tuberculosis", "hepatitis", "malaria", "dengue",
                 "heart disease", "kidney disease", "thyroid", "obesity", "depression", "epilepsy"],
    "treatments": ["surgery", "medication", "therapy", "antibiotics", "vaccine", "insulin", "chemotherapy",
                   "radiation", "physiotherapy", "transplant", "dialysis", "immunotherapy", "inhaler"],
    "tests": ["mri", "ct scan", "x-ray", "blood test", "biopsy", "ecg", "ultrasound", "mammogram", "colonoscopy"],
}
_PATTERNS = {cat: [(t, re.compile(rf"\b{re.escape(t)}s?\b", re.I)) for t in terms]
             for cat, terms in ENTITY_LEXICON.items()}

DISCLAIMER = "This is general information, not a diagnosis. Please consult a doctor about your situation."

SYSTEM = (
    "You are a careful medical information assistant. Use ONLY the MedQuAD context. Explain in plain language, "
    "use short bullet points for symptoms/treatments, and say clearly when the context is insufficient. "
    "Never give personal dosing or diagnose. If the question describes an emergency (chest pain, stroke signs, "
    "severe bleeding, suicidal thoughts), tell the user to contact emergency services first."
)


def recognise_entities(text: str) -> dict[str, list[str]]:
    return {cat: [t for t, p in pats if p.search(text)] for cat, pats in _PATTERNS.items()}


def answer(question: str, kb: KnowledgeBase, k: int = 4) -> dict:
    if not kb.ready:
        raise FileNotFoundError("Medical index not built yet — run: python -m csbot.cli build-medical")
    hits = kb.search(question, k=k, min_score=0.2)
    context = "\n\n---\n\n".join(h.document.text for h in hits) or "(no relevant passages found)"
    text = llm.generate(f"CONTEXT:\n{context}\n\nQUESTION: {question}", system=SYSTEM)
    if DISCLAIMER.split(",")[0].lower() not in text.lower():
        text = f"{text}\n\n_{DISCLAIMER}_"
    return {"answer": text, "entities": recognise_entities(question), "sources": hits}
