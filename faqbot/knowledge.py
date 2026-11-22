"""Retrieval over a small FAQ knowledge base, with two interchangeable backends.

- "embeddings": multilingual sentence embeddings (sentence-transformers). Understands paraphrases
  and other languages, separates off-topic questions better. Needs the model download (~460 MB cache).
- "tfidf": word + character n-gram TF-IDF. No model, no GPU, instant start; used as the fallback.

Optionally an OpenAI-compatible LLM rephrases the retrieved answer, restricted to the retrieved facts.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
# confidence below which the bot hands the question to a human; tuned with evaluate.py per backend.
# embeddings: 0.33-0.48 all give the best score on the eval set (77%), 0.40 is the middle of that range.
DEFAULT_THRESHOLDS = {"tfidf": 0.2, "embeddings": 0.40}


@dataclass
class Match:
    entry_id: str
    question: str
    answer: str
    score: float


class _TfidfEncoder:
    name = "tfidf"

    def __init__(self, docs: list[str]):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.word = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        self.W, self.C = self.word.fit_transform(docs), self.char.fit_transform(docs)

    def similarities(self, query: str) -> np.ndarray:
        from sklearn.metrics.pairwise import cosine_similarity

        w = cosine_similarity(self.word.transform([query]), self.W)[0]
        c = cosine_similarity(self.char.transform([query]), self.C)[0]
        return 0.5 * w + 0.5 * c


class _EmbeddingEncoder:
    name = "embeddings"

    def __init__(self, docs: list[str]):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.D = self.model.encode(docs, normalize_embeddings=True)

    def similarities(self, query: str) -> np.ndarray:
        q = self.model.encode([query], normalize_embeddings=True)[0]
        return self.D @ q


class KnowledgeBase:
    def __init__(self, entries: list[dict], backend: str = "auto", threshold: float | None = None):
        if not entries:
            raise ValueError("Knowledge base is empty")
        self.entries = entries
        docs, self._owner = [], []
        for idx, e in enumerate(entries):
            for text in [e["question"], *e.get("variants", []), e["answer"]]:
                docs.append(text.lower())
                self._owner.append(idx)
        self._owner = np.array(self._owner)
        self.encoder = self._make_encoder(backend, docs)
        self.threshold = threshold if threshold is not None else DEFAULT_THRESHOLDS[self.encoder.name]

    @staticmethod
    def _make_encoder(backend: str, docs: list[str]):
        if backend in ("auto", "embeddings"):
            try:
                return _EmbeddingEncoder(docs)
            except Exception:  # package missing or model not downloadable
                if backend == "embeddings":
                    raise
        return _TfidfEncoder(docs)

    @property
    def backend(self) -> str:
        return self.encoder.name

    @classmethod
    def from_file(cls, path: str | Path, **kwargs) -> "KnowledgeBase":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")), **kwargs)

    def search(self, query: str, top_k: int = 3) -> list[Match]:
        q = query.lower().strip()
        if not q:
            return []
        sims = self.encoder.similarities(q)
        best = np.zeros(len(self.entries))
        np.maximum.at(best, self._owner, sims)  # entry score = its best-matching text
        order = np.argsort(-best)[:top_k]
        return [Match(self.entries[i]["id"], self.entries[i]["question"], self.entries[i]["answer"], float(best[i]))
                for i in order]

    def answer(self, query: str) -> Match | None:
        """Best match, or None when confidence is below the threshold (bot hands over to a human)."""
        matches = self.search(query, top_k=1)
        return matches[0] if matches and matches[0].score >= self.threshold else None


def evaluate(kb: KnowledgeBase, cases: list[dict]) -> dict:
    """Accuracy on labelled questions. expected=null means the bot should NOT answer (off-topic)."""
    misses = []
    for case in cases:
        match = kb.answer(case["q"])
        got = match.entry_id if match else None
        if got != case["expected"]:
            misses.append({"q": case["q"], "expected": case["expected"], "got": got})
    on = [c for c in cases if c["expected"]]
    off = [c for c in cases if not c["expected"]]
    on_miss = sum(1 for m in misses if m["expected"])
    off_miss = len(misses) - on_miss
    return {
        "backend": kb.backend,
        "accuracy": 1 - len(misses) / len(cases),
        "answered_correctly": (len(on) - on_miss) / len(on) if on else None,
        "offtopic_rejected": (len(off) - off_miss) / len(off) if off else None,
        "total": len(cases),
        "misses": misses,
    }


async def rephrase_with_llm(question: str, match: Match) -> str:
    """Optional: let an LLM phrase the answer naturally. Falls back to the stored answer on any error."""
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return match.answer
    base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    prompt = (
        "You are a support assistant of a coffee shop. Answer the customer's question in 1-3 sentences, "
        "in the customer's language, using ONLY the facts below. If the facts do not answer it, say a "
        f"manager will help.\n\nFacts: {match.answer}\n\nCustomer: {question}"
    )
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(f"{base}/chat/completions",
                                     headers={"Authorization": f"Bearer {key}"},
                                     json={"model": model, "temperature": 0.2,
                                           "messages": [{"role": "user", "content": prompt}]})
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
    except (httpx.HTTPError, KeyError, IndexError):
        return match.answer
