"""
sentiment_benchmark.py
-----------------------
Runs the same set of test transcripts through three candidate text-emotion /
sentiment models and logs the results side by side, so a single command
produces the comparison-table evidence for the report:

  1. j-hartmann/emotion-english-distilroberta-base   (Hugging Face, 7-way emotion)
  2. tabularisai/multilingual-sentiment-analysis      (Hugging Face, 5-way sentiment)
  3. VADER                                            (rule-based, no download)

Usage:
    python sentiment_benchmark.py

Each model is optional - if the packages aren't installed, that model is
skipped with a warning rather than crashing the whole run, so you can test
VADER immediately and add the Hugging Face models once installed.

Requires (install what you plan to test):
    pip install vaderSentiment
    pip install transformers torch
"""

import time
import tracemalloc

from test_transcripts import TEST_TRANSCRIPTS


def run_vader():
    """Rule-based baseline. Fast, no model download required."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError:
        print("[VADER] skipped - run: pip install vaderSentiment")
        return []

    analyzer = SentimentIntensityAnalyzer()
    results = []
    for item in TEST_TRANSCRIPTS:
        start = time.perf_counter()
        scores = analyzer.polarity_scores(item["text"])
        latency_ms = (time.perf_counter() - start) * 1000

        compound = scores["compound"]
        if compound >= 0.05:
            label = "positive"
        elif compound <= -0.05:
            label = "negative"
        else:
            label = "neutral"

        results.append({
            "text": item["text"],
            "expected": item["expected"],
            "predicted": label,
            "score": round(compound, 3),
            "latency_ms": round(latency_ms, 2),
        })
    return results


def run_hf_model(model_name, label_key="emotion"):
    """Generic runner for a Hugging Face text-classification pipeline.
    Loads the model once, then times each individual inference call
    separately from the (much slower) one-off load time."""
    try:
        from transformers import pipeline
    except ImportError:
        print(f"[{model_name}] skipped - run: pip install transformers torch")
        return []

    print(f"[{model_name}] loading model (first run downloads the weights)...")
    load_start = time.perf_counter()
    classifier = pipeline("text-classification", model=model_name, top_k=1)
    load_time_s = time.perf_counter() - load_start
    print(f"[{model_name}] loaded in {load_time_s:.1f}s")

    results = []
    for item in TEST_TRANSCRIPTS:
        start = time.perf_counter()
        output = classifier(item["text"])
        latency_ms = (time.perf_counter() - start) * 1000

        # top_k=1 returns a nested list like [[{"label": ..., "score": ...}]]
        top = output[0][0] if isinstance(output[0], list) else output[0]

        results.append({
            "text": item["text"],
            "expected": item["expected"],
            "predicted": top["label"],
            "score": round(top["score"], 3),
            "latency_ms": round(latency_ms, 2),
        })
    return results


def print_table(model_name, results):
    if not results:
        return
    print(f"\n=== {model_name} ===")
    print(f"{'expected':>10} | {'predicted':>10} | {'score':>6} | {'ms':>7} | text")
    print("-" * 90)
    for r in results:
        text_preview = r["text"][:45] + ("..." if len(r["text"]) > 45 else "")
        print(f"{r['expected']:>10} | {r['predicted']:>10} | {r['score']:>6} | "
              f"{r['latency_ms']:>7} | {text_preview}")
    avg_latency = sum(r["latency_ms"] for r in results) / len(results)
    print(f"\nAverage latency: {avg_latency:.2f} ms/input over {len(results)} inputs")


if __name__ == "__main__":
    tracemalloc.start()

    vader_results = run_vader()
    print_table("VADER", vader_results)

    hartmann_results = run_hf_model("j-hartmann/emotion-english-distilroberta-base")
    print_table("j-hartmann/emotion-english-distilroberta-base", hartmann_results)

    tabularisai_results = run_hf_model("tabularisai/multilingual-sentiment-analysis")
    print_table("tabularisai/multilingual-sentiment-analysis", tabularisai_results)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"\nPeak Python memory during run: {peak / 1024 / 1024:.1f} MB")
    print("(Note: this does not include memory used inside PyTorch/C extensions -")
    print(" for a fuller picture, watch Task Manager/Activity Monitor while running.)")
