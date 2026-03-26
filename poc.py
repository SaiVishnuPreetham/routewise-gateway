#!/usr/bin/env python3
"""
RouteWise — Proof of Concept (PoC)
====================================
Standalone evaluator for the routing model.

This script demonstrates the routing model in isolation:
  - Loads a JSON or CSV file of labeled prompts
  - Runs each prompt through the routing model
  - Compares decisions against ground-truth labels
  - Prints per-prompt results and a summary

Requirements
------------
  - Python 3.10+
  - No API keys, no running server, no LiteLLM dependency at runtime

Usage
-----
    python poc.py                          # uses default test_suite.json
    python poc.py test_suite.json          # explicit file path
    python poc.py my_prompts.csv           # CSV with columns: prompt, ground_truth
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

# Import the routing model — the ONLY dependency
from gateway.routing_model import RoutingModel, FAST_LABEL, CAPABLE_LABEL

# ---------------------------------------------------------------------------
# Mapping: ground_truth label → expected routing decision
# ---------------------------------------------------------------------------
#   "simple"  → should route to Fast model
#   "complex" → should route to Capable model

LABEL_TO_ROUTE = {
    "simple": FAST_LABEL,
    "complex": CAPABLE_LABEL,
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_json(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def load_csv(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=1):
            rows.append({
                "id": row.get("id", i),
                "prompt": row["prompt"],
                "ground_truth": row["ground_truth"].strip().lower(),
            })
    return rows


def load_test_suite(path: Path) -> list[dict]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return load_json(path)
    elif suffix == ".csv":
        return load_csv(path)
    else:
        print(f"ERROR: Unsupported file format '{suffix}'. Use .json or .csv")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run_evaluation(prompts: list[dict], model: RoutingModel) -> dict:
    """
    Run each prompt through the routing model and compare to ground truth.

    Returns a summary dict with accuracy, FP rate, FN rate, and per-prompt results.
    """
    results = []
    correct = 0
    false_positives = 0   # complex prompt sent to Fast (under-routed)
    false_negatives = 0   # simple prompt sent to Capable (over-routed)

    total_latency_ms = 0.0

    for item in prompts:
        prompt = item["prompt"]
        gt = item["ground_truth"].strip().lower()
        expected_route = LABEL_TO_ROUTE.get(gt)

        if expected_route is None:
            print(f"WARNING: Unknown ground_truth '{gt}' for prompt id={item.get('id')}")
            continue

        decision = model.classify(prompt)
        is_correct = decision.label == expected_route

        if is_correct:
            correct += 1
        else:
            if gt == "complex" and decision.label == FAST_LABEL:
                false_positives += 1      # complex → Fast (dangerous)
            elif gt == "simple" and decision.label == CAPABLE_LABEL:
                false_negatives += 1      # simple → Capable (wasteful)

        total_latency_ms += decision.latency_ms

        results.append({
            "id": item.get("id", "?"),
            "prompt_snippet": prompt[:70] + ("..." if len(prompt) > 70 else ""),
            "ground_truth": gt,
            "predicted": decision.label,
            "correct": "Y" if is_correct else "N",
            "confidence": decision.confidence,
            "raw_score": decision.raw_score,
            "reason": decision.reason,
            "latency_ms": decision.latency_ms,
        })

    total = len(results)
    accuracy = correct / total if total > 0 else 0.0
    fp_rate = false_positives / total if total > 0 else 0.0
    fn_rate = false_negatives / total if total > 0 else 0.0
    avg_latency = total_latency_ms / total if total > 0 else 0.0

    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "false_positives": false_positives,
        "false_positive_rate": fp_rate,
        "false_negatives": false_negatives,
        "false_negative_rate": fn_rate,
        "avg_latency_ms": avg_latency,
        "results": results,
    }


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def print_results(summary: dict) -> None:
    results = summary["results"]

    print("=" * 100)
    print("  RouteWise PoC -- Routing Model Evaluation")
    print("=" * 100)
    print()

    # Per-prompt table
    header = f"{'ID':>4} {'Correct':>7} {'GT':>8} {'Predicted':>9} {'Conf':>6} {'Score':>6} {'Latency':>8}  Prompt"
    print(header)
    print("-" * 100)

    for r in results:
        print(
            f"{r['id']:>4} "
            f"{r['correct']:>7} "
            f"{r['ground_truth']:>8} "
            f"{r['predicted']:>9} "
            f"{r['confidence']:>6.2f} "
            f"{r['raw_score']:>6.3f} "
            f"{r['latency_ms']:>7.1f}ms"
            f"  {r['prompt_snippet']}"
        )

    # Summary
    print()
    print("=" * 100)
    print("  SUMMARY")
    print("=" * 100)
    print(f"  Total prompts evaluated  : {summary['total']}")
    print(f"  Correct predictions      : {summary['correct']} / {summary['total']}")
    print(f"  Overall Accuracy         : {summary['accuracy']:.1%}")
    print(f"  False Positives (FP)     : {summary['false_positives']}  (complex -> Fast model = under-routed)")
    print(f"  False Positive Rate      : {summary['false_positive_rate']:.1%}")
    print(f"  False Negatives (FN)     : {summary['false_negatives']}  (simple -> Capable model = over-routed)")
    print(f"  False Negative Rate      : {summary['false_negative_rate']:.1%}")
    print(f"  Avg Routing Latency      : {summary['avg_latency_ms']:.2f} ms")
    print()

    # Success bar check
    target_accuracy = 0.75
    if summary["accuracy"] >= target_accuracy:
        print(f"  [PASS] Accuracy {summary['accuracy']:.1%} meets the >75% success bar")
    else:
        print(f"  [FAIL] Accuracy {summary['accuracy']:.1%} is below the >75% success bar")

    print()

    # Show mis-routes for failure analysis
    mis_routes = [r for r in results if r["correct"] == "N"]
    if mis_routes:
        print("-" * 100)
        print("  FAILURE ANALYSIS -- Mis-routed Prompts")
        print("-" * 100)
        for r in mis_routes:
            print(f"  ID {r['id']}: GT={r['ground_truth']}, Predicted={r['predicted']}, Score={r['raw_score']:.3f}")
            print(f"    Reason: {r['reason']}")
            print(f"    Prompt: {r['prompt_snippet']}")
            print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Determine file path
    if len(sys.argv) > 1:
        file_path = Path(sys.argv[1])
    else:
        file_path = Path(__file__).parent / "test_suite.json"

    if not file_path.exists():
        print(f"ERROR: File not found: {file_path}")
        sys.exit(1)

    print(f"Loading test suite from: {file_path}")

    # Load data
    prompts = load_test_suite(file_path)
    print(f"Loaded {len(prompts)} prompts")
    print()

    # Run evaluation
    model = RoutingModel()
    t0 = time.perf_counter()
    summary = run_evaluation(prompts, model)
    total_time = time.perf_counter() - t0

    # Print results
    print_results(summary)
    print(f"  Total evaluation time: {total_time:.2f}s")
    print()


if __name__ == "__main__":
    main()
