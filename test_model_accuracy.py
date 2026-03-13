"""
test_model_accuracy.py
======================
Tests the Show / Tell / Not a sentence classification accuracy
of different OpenRouter models and saves results to CSV files.

Usage:
    python test_model_accuracy.py

Outputs:
    results/accuracy_per_sentence.csv  – one row per (model, sentence)
    results/accuracy_summary.csv       – one row per model with aggregate metrics
"""

import os
import re
import json
import time
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Read API key from .streamlit/secrets.toml
def _read_openrouter_key() -> str:
    secrets_path = Path(".streamlit/secrets.toml")
    if secrets_path.exists():
        for line in secrets_path.read_text().splitlines():
            if line.strip().startswith("OPENROUTER_API_KEY"):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return os.environ.get("OPENROUTER_API_KEY", "")

OPENROUTER_API_KEY = _read_openrouter_key()

# Models to benchmark
MODELS = [
    "nvidia/nemotron-3-nano-30b-a3b",
    "qwen/qwen3-next-80b-a3b-instruct",
    "openrouter/hunter-alpha",
    "google/gemini-3.1-flash-lite-preview",
    "stepfun/step-3.5-flash",
    "deepseek/deepseek-v3.2",
    "openai/gpt-4o-mini",
    "xiaomi/mimo-v2-flash",
    "x-ai/grok-4-fast",
    "meta-llama/llama-3.3-70b-instruct"
]



# Output directory
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Labeled test dataset loaded from test_dataset_balanced.csv
# Uses sentence_text as input and groundtruth as label.
# Rows where groundtruth is missing are excluded.
# ---------------------------------------------------------------------------

def _load_test_data(csv_path: str = "test_dataset_balanced.csv") -> list[dict]:
    df = pd.read_csv(csv_path)
    df = df[df["groundtruth"].notna() & (df["groundtruth"] != "")]
    return [
        {"sentence": row["sentence_text"], "true_label": row["groundtruth"]}
        for _, row in df.iterrows()
        if str(row["sentence_text"]).strip()
    ]

TEST_DATA = _load_test_data()

# ---------------------------------------------------------------------------
# Classification (same logic as streamlit_predict_app.py)
# ---------------------------------------------------------------------------

def call_openrouter(prompt: str, api_key: str, model: str) -> tuple[str, dict]:
    """Returns (content_text, usage_dict) where usage_dict has prompt_tokens and completion_tokens."""
    resp = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=60,
    )
    if not resp.ok:
        print(f"    [DEBUG] Status: {resp.status_code}, Body: {resp.text[:500]}")
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage") or {}
    return content, {
        "prompt_tokens":     int(usage.get("prompt_tokens", 0)),
        "completion_tokens": int(usage.get("completion_tokens", 0)),
    }


def classify(sentences: list[str], api_key: str, model: str) -> list[str]:
    """Returns a list of labels: 'Show', 'Tell', or 'Not a sentence'."""
    numbered = "\n".join(f'{i+1}. "{s}"' for i, s in enumerate(sentences))
    prompt = (
        'You are classifying items from student data stories as "Show", "Tell", or "Not a sentence".\n\n'
        "Definitions:\n"
        '- "Show" sentences are DESCRIPTIVE – they describe what is literally visible in the data/chart.\n'
        '- "Tell" sentences are INTERPRETIVE – they make claims, draw conclusions, or interpret beyond what\'s directly visible.\n'
        '- "Not a sentence" items are fragments, titles, headings, labels, or any text that is not a grammatically complete sentence.\n\n'
        "Classify each item below. Return ONLY a JSON array of labels, one per item, "
        'where each label is exactly "Show", "Tell", or "Not a sentence".\n'
        'Example output for 4 items: ["Show", "Tell", "Not a sentence", "Show"]\n\n'
        f"Items:\n{numbered}"
    )
    try:
        raw, _ = call_openrouter(prompt, api_key, model)
        match = re.search(r"\[.*?\]", raw, re.DOTALL)
        if match:
            labels = json.loads(match.group())
            if len(labels) == len(sentences):
                def _norm(l):
                    s = str(l).strip().lower()
                    if s == "show": return "Show"
                    if s == "not a sentence": return "Not a sentence"
                    return "Tell"
                return [_norm(l) for l in labels]
    except Exception as e:
        print(f"    [ERROR] {e}")
    # fallback
    return ["Tell"] * len(sentences)


# ---------------------------------------------------------------------------
# Single-sentence classification
# ---------------------------------------------------------------------------

SINGLE_SENTENCE_PROMPT_TEMPLATE = (
    'You are classifying a sentence from a student data story.\n\n'
    'Definitions:\n'
    '- "Show" – DESCRIPTIVE: describes what is literally visible in the data or chart.\n'
    '- "Tell" – INTERPRETIVE: makes a claim, draws a conclusion, or interprets beyond what is directly visible.\n'
    '- "Not a sentence" – a fragment, title, heading, label, or any text that is not a grammatically complete sentence.\n\n'
    'Reply with EXACTLY ONE of these three labels and nothing else: Show | Tell | Not a sentence\n\n'
    'Sentence: "{sentence}"'
)


def _parse_single_label(raw: str) -> str:
    lower = raw.strip().lower()
    if "not a sentence" in lower:
        return "Not a sentence"
    if "show" in lower:
        return "Show"
    if "tell" in lower:
        return "Tell"
    return "Tell"


def classify_one(sentence: str, api_key: str, model: str) -> tuple[str, dict]:
    """Classify a single sentence. Returns (label, usage_dict)."""
    prompt = SINGLE_SENTENCE_PROMPT_TEMPLATE.format(sentence=sentence)
    try:
        content, usage = call_openrouter(prompt, api_key, model)
        return _parse_single_label(content), usage
    except Exception as e:
        print(f"    [ERROR] {e}")
        return "Tell", {"prompt_tokens": 0, "completion_tokens": 0}


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

LABELS = ["Show", "Tell", "Not a sentence"]

def compute_metrics(true_labels: list[str], pred_labels: list[str]) -> dict:
    total = len(true_labels)
    correct = sum(t == p for t, p in zip(true_labels, pred_labels))
    accuracy = correct / total if total else 0.0

    metrics = {"accuracy": round(accuracy, 4), "total": total, "correct": correct}

    for cls in LABELS:
        tp = sum(t == cls and p == cls for t, p in zip(true_labels, pred_labels))
        fp = sum(t != cls and p == cls for t, p in zip(true_labels, pred_labels))
        fn = sum(t == cls and p != cls for t, p in zip(true_labels, pred_labels))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall    = tp / (tp + fn) if (tp + fn) else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        key = cls.lower().replace(" ", "_")
        metrics[f"{key}_precision"] = round(precision, 4)
        metrics[f"{key}_recall"]    = round(recall, 4)
        metrics[f"{key}_f1"]        = round(f1, 4)

    macro_f1 = sum(metrics[f"{cls.lower().replace(' ', '_')}_f1"] for cls in LABELS) / len(LABELS)
    metrics["macro_f1"] = round(macro_f1, 4)
    return metrics


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    if not OPENROUTER_API_KEY:
        print("ERROR: OPENROUTER_API_KEY not found. Set it in .streamlit/secrets.toml or as env var.")
        return

    sentences   = [d["sentence"]   for d in TEST_DATA]
    true_labels = [d["true_label"] for d in TEST_DATA]
    n           = len(sentences)

    per_sentence_rows = []
    summary_rows      = []

    for model in MODELS:
        print(f"\n{'='*60}")
        print(f"Model: {model}")
        print(f"{'='*60}")

        pred_labels      = []
        total_prompt_tok = 0
        total_compl_tok  = 0

        for idx, sentence in enumerate(sentences, start=1):
            print(f"  sentence {idx}/{n}...", end="\r", flush=True)
            label, usage = classify_one(sentence, OPENROUTER_API_KEY, model)
            pred_labels.append(label)
            total_prompt_tok += usage["prompt_tokens"]
            total_compl_tok  += usage["completion_tokens"]
            time.sleep(0.3)

        print()  # newline after \r progress line

        for sent, true, pred in zip(sentences, true_labels, pred_labels):
            correct = true == pred
            print(f"  {'✓' if correct else '✗'}  [{true:18s}] → [{pred:18s}]  \"{sent[:60]}\"")
            per_sentence_rows.append({
                "model":      model,
                "sentence":   sent,
                "true_label": true,
                "pred_label": pred,
                "correct":    int(correct),
            })

        m = compute_metrics(true_labels, pred_labels)
        print(f"\n  Accuracy: {m['accuracy']:.1%}  |  Macro F1: {m['macro_f1']:.4f}")
        print(f"  Show     — P:{m['show_precision']:.2f}  R:{m['show_recall']:.2f}  F1:{m['show_f1']:.2f}")
        print(f"  Tell     — P:{m['tell_precision']:.2f}  R:{m['tell_recall']:.2f}  F1:{m['tell_f1']:.2f}")
        print(f"  Not sent — P:{m['not_a_sentence_precision']:.2f}  R:{m['not_a_sentence_recall']:.2f}  F1:{m['not_a_sentence_f1']:.2f}")

        print(f"\n  Tokens — prompt: {total_prompt_tok:,}  completion: {total_compl_tok:,}")

        summary_rows.append({
            "model":             model,
            "prompt_tokens":     total_prompt_tok,
            "completion_tokens": total_compl_tok,
            **m,
        })

    # -----------------------------------------------------------------------
    # Final summary table
    # -----------------------------------------------------------------------
    col_w = 40
    print(f"\n{'='*90}")
    print(f"{'FINAL SUMMARY':^90}")
    print(f"{'='*90}")
    print(f"{'Model':<{col_w}} {'Accuracy':>9} {'Macro F1':>9} {'Show F1':>8} {'Tell F1':>8} {'Correct':>8}")
    print(f"{'-'*90}")
    for r in summary_rows:
        print(
            f"{r['model']:<{col_w}} "
            f"{r['accuracy']:>8.1%} "
            f"{r['macro_f1']:>9.4f} "
            f"{r['show_f1']:>8.4f} "
            f"{r['tell_f1']:>8.4f} "
            f"{r['correct']:>5}/{r['total']} "
        )
    print(f"{'='*105}")


if __name__ == "__main__":
    main()
