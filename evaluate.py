"""Measure how well the bot answers: python evaluate.py [--backend tfidf|embeddings] [--threshold 0.3]"""
import argparse
import json
from pathlib import Path

from faqbot.knowledge import KnowledgeBase, evaluate

ROOT = Path(__file__).parent

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="auto", choices=["auto", "tfidf", "embeddings"])
    ap.add_argument("--threshold", type=float)
    args = ap.parse_args()

    kb = KnowledgeBase.from_file(ROOT / "data" / "knowledge_base.json", backend=args.backend, threshold=args.threshold)
    cases = json.loads((ROOT / "data" / "eval_questions.json").read_text(encoding="utf-8"))
    r = evaluate(kb, cases)
    print(f"Backend: {r['backend']} (threshold {kb.threshold})")
    print(f"Overall accuracy:          {r['accuracy']:.0%} on {r['total']} questions")
    print(f"Correct answer found:      {r['answered_correctly']:.0%}")
    print(f"Off-topic handed to human: {r['offtopic_rejected']:.0%}")
    for m in r["misses"]:
        print(f"  MISS: {m['q']!r} expected={m['expected']} got={m['got']}")
