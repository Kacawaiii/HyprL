#!/usr/bin/env python3
"""Score the cached news with FinBERT and write a scored parquet per month.

Why FinBERT and not the alternatives — measured, not assumed (see session notes):
  headline: "Q3 earnings beat estimates, guidance raised"
    FinBERT +0.96   LM 0.00   VADER 0.00
  headline: "Vice President of Crude Oil Operations announces record production"
    FinBERT  0.00   LM 0.00   VADER -0.57   <- 'vice'/'crude' trap
  headline: "higher tax liability and increased cost of capital"
    FinBERT -0.69   LM 0.00   VADER +0.08   <- VADER gets the sign backwards
VADER is trained on social media; Loughran-McDonald is built from 10-K filings and
is missing the entire bullish news vocabulary (BEAT/TOPS/RAISES/UPGRADE/SURGES all
absent, while MISS/DOWNGRADE/WARNS are present) — applied to headlines it is
systematically bearish-biased. FinBERT is fine-tuned on financial news, which is
what we actually have.

Scoring is cached per month because it is the expensive step; the IC study reruns
against the cache.
"""
from __future__ import annotations
import sys
import time
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/news/raw"
OUT = ROOT / "data/news/scored"
OUT.mkdir(parents=True, exist_ok=True)

BATCH = 64
# Headlines run ~12 words (~20 tokens) and this feed's `summary` is almost always
# empty, so 32 covers them. Attention is O(n^2) in sequence length, so this is the
# cheapest real speedup available.
MAXLEN = 32


def main():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch

    name = "ProsusAI/finbert"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name)
    model.eval()

    # CPU throughput: fp32 BERT-base ran ~28 articles/s at 6 threads / maxlen 128,
    # i.e. ~6h for the full history. Using every core plus maxlen 32 gets ~162/s
    # (~1h) — enough, with no change to the model's output.
    #
    # int8 dynamic quantization was tried and REJECTED: measured, it mangles the
    # negative tail ("SEC investigation ... fraud" went -0.91 -> -0.19, error 0.73)
    # for only a 1.4x gain. Destroying the bearish signal to save 20 minutes is a
    # bad trade — bad news is most of what this feed carries.
    import os as _os
    torch.set_num_threads(_os.cpu_count() or 4)
    print(f"fp32, threads={torch.get_num_threads()}, maxlen={MAXLEN}")
    # label order from the model config, not hardcoded
    id2label = {i: l.lower() for i, l in model.config.id2label.items()}
    sign = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
    w = torch.tensor([sign[id2label[i]] for i in range(len(id2label))])

    files = sorted(RAW.glob("news_*.parquet"))
    if not files:
        sys.exit("no cached news — run fetch_news.py first")

    for f in files:
        dst = OUT / f.name
        if dst.exists():
            print(f"{f.stem}: cached")
            continue
        df = pd.read_parquet(f)
        texts = (df["headline"].fillna("") + ". " + df["summary"].fillna("")).tolist()
        t0, scores = time.time(), []
        with torch.no_grad():
            for i in range(0, len(texts), BATCH):
                b = tok(texts[i:i + BATCH], padding=True, truncation=True,
                        max_length=MAXLEN, return_tensors="pt")
                p = torch.softmax(model(**b).logits, dim=-1)
                # expected sentiment = P(pos) - P(neg); neutral contributes 0.
                # Keeps the confidence information instead of collapsing to argmax.
                scores.extend((p @ w).tolist())
        df["finbert"] = scores
        df.to_parquet(dst)
        rate = len(texts) / max(1e-9, time.time() - t0)
        print(f"{f.stem}: {len(texts)} scored in {time.time()-t0:.0f}s ({rate:.0f}/s)")
    print(f"done -> {OUT}")


if __name__ == "__main__":
    main()
