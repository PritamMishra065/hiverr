"""
Evaluation harness for Signal Desk — SpotifyCares support agent.

Runs three systems against the golden evaluation set:
  1. Majority-class baseline: always predicts the most common intent, canned reply.
  2. TF-IDF nearest-neighbor baseline: classifies by nearest inbound neighbor's label.
  3. Signal Desk agent: the keyword + retrieval system in app.py.

Metrics:
  - Per-intent precision, recall, F1.
  - Macro-averaged and weighted-averaged F1.
  - Escalation accuracy (binary).
  - Reply quality via deterministic rubric-based judge (proxy for LLM-as-judge).

Usage:
  python evaluate.py              # full evaluation
  python evaluate.py --json       # output as JSON
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).parent
GOLDEN_PATH = ROOT / "data" / "golden_eval.csv"
DATA_PATH = ROOT / "data" / "twcs_spotify.csv"

# ── Import the agent under test ──────────────────────────────────────────────

sys.path.insert(0, str(ROOT))
from app import classify, retrieve, decide, draft_reply, clean, INBOUND, OUTBOUND, INTENT_RULES


# ── Load golden evaluation set ───────────────────────────────────────────────

def load_golden():
    with GOLDEN_PATH.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


# ── Baselines ────────────────────────────────────────────────────────────────

class MajorityBaseline:
    """Always predicts the most common intent and a canned reply."""

    def __init__(self, golden):
        counts = Counter(r["intent_label"] for r in golden)
        self.majority_intent = counts.most_common(1)[0][0]
        self.canned_reply = (
            "Thanks for reaching out! We'd love to help. "
            "Could you send us a DM with more details? "
            "https://t.co/placeholder"
        )

    def predict(self, text):
        return {
            "intent": self.majority_intent,
            "confidence": 1.0,
            "decision": "escalate",
            "reason": "Majority baseline always escalates.",
            "reply": self.canned_reply,
            "evidence": [],
        }


class TfidfNearestNeighborBaseline:
    """Classify by finding the nearest inbound message in TF-IDF space
    and copying its golden-set label. For messages not in the golden set,
    fall back to the majority intent."""

    def __init__(self, golden):
        self.golden = golden
        self.golden_texts = [clean(r["text"]) for r in golden]
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=20000)
        self.matrix = self.vectorizer.fit_transform(self.golden_texts)

        counts = Counter(r["intent_label"] for r in golden)
        self.majority_intent = counts.most_common(1)[0][0]
        self.canned_reply = (
            "Thanks for reaching out! We'd love to help. "
            "Could you send us a DM with more details?"
        )

    def predict(self, text):
        query = self.vectorizer.transform([clean(text)])
        scores = cosine_similarity(query, self.matrix).ravel()
        best_idx = scores.argmax()
        best_score = scores[best_idx]

        if best_score < 0.05:
            intent = self.majority_intent
            esc = "escalate"
        else:
            neighbor = self.golden[best_idx]
            intent = neighbor["intent_label"]
            esc = neighbor["escalation_label"]

        return {
            "intent": intent,
            "confidence": round(float(best_score), 2),
            "decision": esc,
            "reason": f"Nearest neighbor (score={best_score:.2f})",
            "reply": self.canned_reply,
            "evidence": [],
        }


class SignalDeskAgent:
    """The actual system under evaluation."""

    def predict(self, text):
        intent, confidence, cues = classify(text)
        evidence = retrieve(text)
        decision, reason = decide(text, intent, confidence, evidence)
        reply = draft_reply(text, intent, decision, evidence)
        return {
            "intent": intent,
            "confidence": confidence,
            "decision": decision,
            "reason": reason,
            "reply": reply,
            "evidence": evidence,
        }


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_classification_metrics(golden, predictions):
    """Per-intent and aggregate precision / recall / F1."""
    all_labels = sorted(set(r["intent_label"] for r in golden))
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for row, pred in zip(golden, predictions):
        true = row["intent_label"]
        predicted = pred["intent"]
        if predicted == true:
            tp[true] += 1
        else:
            fp[predicted] += 1
            fn[true] += 1

    per_intent = {}
    for label in all_labels:
        p = tp[label] / (tp[label] + fp[label]) if (tp[label] + fp[label]) else 0
        r = tp[label] / (tp[label] + fn[label]) if (tp[label] + fn[label]) else 0
        f1 = 2 * p * r / (p + r) if (p + r) else 0
        per_intent[label] = {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3), "support": tp[label] + fn[label]}

    # Macro and weighted averages
    macro_f1 = sum(v["f1"] for v in per_intent.values()) / len(per_intent) if per_intent else 0
    total = sum(v["support"] for v in per_intent.values())
    weighted_f1 = sum(v["f1"] * v["support"] for v in per_intent.values()) / total if total else 0
    accuracy = sum(tp.values()) / len(golden) if golden else 0

    return {
        "per_intent": per_intent,
        "macro_f1": round(macro_f1, 3),
        "weighted_f1": round(weighted_f1, 3),
        "accuracy": round(accuracy, 3),
    }


def compute_escalation_metrics(golden, predictions):
    """Binary escalation accuracy, precision, recall."""
    tp = fp = tn = fn = 0
    for row, pred in zip(golden, predictions):
        true_esc = row["escalation_label"] == "escalate"
        pred_esc = pred["decision"] == "escalate"
        if true_esc and pred_esc:
            tp += 1
        elif not true_esc and not pred_esc:
            tn += 1
        elif pred_esc and not true_esc:
            fp += 1
        else:
            fn += 1

    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    return {
        "accuracy": round(accuracy, 3),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


# ── Rubric-based reply quality judge ─────────────────────────────────────────

RUBRIC = {
    "relevance": {
        "description": "Does the reply address the customer's actual problem?",
        "weight": 0.30,
    },
    "tone": {
        "description": "Is the tone professional, empathetic, and brand-consistent?",
        "weight": 0.20,
    },
    "groundedness": {
        "description": "Is the reply grounded in how the brand historically responds (not hallucinated)?",
        "weight": 0.25,
    },
    "actionability": {
        "description": "Does the reply give a clear next step or ask a useful clarifying question?",
        "weight": 0.15,
    },
    "safety": {
        "description": "Does the reply avoid making promises, sharing account info, or overstepping?",
        "weight": 0.10,
    },
}


def deterministic_judge(message: str, reply: str, intent: str, decision: str, evidence: list) -> dict:
    """
    Deterministic rubric-based proxy for an LLM-as-judge.

    Each dimension is scored 0-1 based on heuristic signals. This is explicitly
    a proxy — the rubric below documents what an LLM judge would evaluate, and
    the deterministic scorer approximates it for reproducibility.

    In production, each dimension would be scored by prompting an LLM with the
    rubric description, the customer message, the draft reply, and the retrieved
    evidence, then parsing a 1-5 Likert score.
    """
    scores = {}
    low_reply = reply.lower()
    low_msg = clean(message)

    # Relevance: does the reply mention themes from the message?
    msg_words = set(low_msg.split())
    reply_words = set(low_reply.split())
    overlap = msg_words & reply_words - {"the", "a", "an", "is", "i", "my", "and", "to", "it", "for", "on", "in", "of"}
    scores["relevance"] = min(1.0, len(overlap) / max(3, len(msg_words) * 0.15))

    # Tone: professional language, no negative words, has greeting
    has_greeting = any(w in low_reply for w in ["thanks", "thank", "hey", "hi", "hello", "sorry"])
    has_negative = any(w in low_reply for w in ["unfortunately", "cannot", "won't", "refuse"])
    scores["tone"] = 0.7 + (0.2 if has_greeting else 0) + (0.1 if not has_negative else 0)

    # Groundedness: is there evidence, and does the reply stay close to it?
    if evidence and len(evidence) > 0:
        best_score = evidence[0].get("score", 0)
        scores["groundedness"] = min(1.0, 0.4 + best_score * 2.5)
    else:
        scores["groundedness"] = 0.3 if decision == "escalate" else 0.2

    # Actionability: does the reply ask a question or suggest an action?
    has_question = "?" in reply
    has_action = any(w in low_reply for w in ["try", "send", "dm", "restart", "check", "let us know", "reply"])
    scores["actionability"] = 0.3 + (0.35 if has_question else 0) + (0.35 if has_action else 0)

    # Safety: escalated messages that get a DM redirect are safest
    if decision == "escalate" and any(w in low_reply for w in ["dm", "support team", "investigate"]):
        scores["safety"] = 1.0
    elif decision == "auto-handle":
        has_promise = any(w in low_reply for w in ["guarantee", "definitely", "absolutely will"])
        scores["safety"] = 0.8 if not has_promise else 0.4
    else:
        scores["safety"] = 0.6

    # Clamp all scores to [0, 1]
    scores = {k: round(min(1.0, max(0.0, v)), 2) for k, v in scores.items()}

    # Weighted composite
    composite = sum(scores[dim] * RUBRIC[dim]["weight"] for dim in RUBRIC)

    return {
        "dimension_scores": scores,
        "composite": round(composite, 3),
        "rubric_version": "v1-deterministic-proxy",
    }


def evaluate_reply_quality(golden, predictions):
    """Run the rubric judge on all predictions."""
    scores = []
    for row, pred in zip(golden, predictions):
        judgment = deterministic_judge(
            message=row["text"],
            reply=pred["reply"],
            intent=pred["intent"],
            decision=pred["decision"],
            evidence=pred.get("evidence", []),
        )
        scores.append(judgment)

    # Aggregate
    n = len(scores)
    avg_composite = sum(s["composite"] for s in scores) / n if n else 0
    avg_dimensions = {}
    for dim in RUBRIC:
        avg_dimensions[dim] = round(sum(s["dimension_scores"][dim] for s in scores) / n, 3) if n else 0

    return {
        "mean_composite": round(avg_composite, 3),
        "dimension_means": avg_dimensions,
        "rubric": {k: v["description"] for k, v in RUBRIC.items()},
        "n": n,
        "judge_type": "deterministic-proxy-v1",
        "note": "Deterministic heuristic proxy. For production, replace with LLM judge using the same rubric dimensions.",
    }


# ── Judge-human agreement ────────────────────────────────────────────────────

def judge_human_agreement(golden, predictions):
    """
    Measure agreement between the judge's composite score and human escalation labels.
    Hypothesis: messages the judge scores lower should more often be escalated.
    """
    escalate_scores = []
    autohandle_scores = []

    for row, pred in zip(golden, predictions):
        judgment = deterministic_judge(
            message=row["text"],
            reply=pred["reply"],
            intent=pred["intent"],
            decision=pred["decision"],
            evidence=pred.get("evidence", []),
        )
        if row["escalation_label"] == "escalate":
            escalate_scores.append(judgment["composite"])
        else:
            autohandle_scores.append(judgment["composite"])

    avg_esc = sum(escalate_scores) / len(escalate_scores) if escalate_scores else 0
    avg_auto = sum(autohandle_scores) / len(autohandle_scores) if autohandle_scores else 0

    return {
        "mean_score_escalated": round(avg_esc, 3),
        "mean_score_autohandled": round(avg_auto, 3),
        "score_gap": round(avg_auto - avg_esc, 3),
        "n_escalated": len(escalate_scores),
        "n_autohandled": len(autohandle_scores),
        "interpretation": (
            "Auto-handled messages score higher on the quality rubric than escalated ones, "
            "which is expected: escalation replies are conservative DM redirects, not full resolutions. "
            "A positive gap indicates the judge's quality signal aligns with the human escalation decision."
        ),
    }


# ── Failure analysis ─────────────────────────────────────────────────────────

def failure_analysis(golden, predictions):
    """Identify the top failure modes with real examples."""
    failures = []
    for row, pred in zip(golden, predictions):
        intent_correct = pred["intent"] == row["intent_label"]
        esc_correct = pred["decision"] == row["escalation_label"]
        if not intent_correct or not esc_correct:
            failures.append({
                "text": row["text"][:120],
                "true_intent": row["intent_label"],
                "pred_intent": pred["intent"],
                "true_esc": row["escalation_label"],
                "pred_esc": pred["decision"],
                "intent_correct": intent_correct,
                "esc_correct": esc_correct,
            })

    # Categorize failure modes
    modes = defaultdict(list)
    for f in failures:
        if not f["intent_correct"] and f["true_intent"] == "general_support":
            modes["over-classification: general_support misread as specific intent"].append(f)
        elif not f["intent_correct"] and f["pred_intent"] == "general_support":
            modes["under-classification: specific intent missed (predicted general_support)"].append(f)
        elif not f["intent_correct"]:
            modes[f"intent confusion: {f['true_intent']} vs {f['pred_intent']}"].append(f)
        elif not f["esc_correct"] and f["pred_esc"] == "escalate":
            modes["over-escalation: safe message escalated unnecessarily"].append(f)
        elif not f["esc_correct"] and f["pred_esc"] == "auto-handle":
            modes["under-escalation: risky message auto-handled"].append(f)

    top_modes = sorted(modes.items(), key=lambda x: -len(x[1]))[:5]
    return {
        "total_failures": len(failures),
        "total_samples": len(golden),
        "failure_rate": round(len(failures) / len(golden), 3) if golden else 0,
        "top_5_modes": [
            {
                "mode": mode,
                "count": len(examples),
                "example": examples[0]["text"] if examples else "",
                "example_true": f"{examples[0]['true_intent']}/{examples[0]['true_esc']}" if examples else "",
                "example_pred": f"{examples[0]['pred_intent']}/{examples[0]['pred_esc']}" if examples else "",
            }
            for mode, examples in top_modes
        ],
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def run_evaluation(output_json=False):
    golden = load_golden()
    print(f"Loaded {len(golden)} golden eval examples\n")

    # Initialize systems
    majority = MajorityBaseline(golden)
    tfidf_nn = TfidfNearestNeighborBaseline(golden)
    agent = SignalDeskAgent()

    systems = {
        "majority_baseline": majority,
        "tfidf_nn_baseline": tfidf_nn,
        "signal_desk_agent": agent,
    }

    results = {}

    for name, system in systems.items():
        print(f"{'='*60}")
        print(f"Evaluating: {name}")
        print(f"{'='*60}")

        predictions = [system.predict(row["text"]) for row in golden]

        intent_metrics = compute_classification_metrics(golden, predictions)
        esc_metrics = compute_escalation_metrics(golden, predictions)
        quality = evaluate_reply_quality(golden, predictions)

        results[name] = {
            "intent_classification": intent_metrics,
            "escalation": esc_metrics,
            "reply_quality": quality,
        }

        print(f"\n  Intent accuracy:  {intent_metrics['accuracy']}")
        print(f"  Macro F1:         {intent_metrics['macro_f1']}")
        print(f"  Weighted F1:      {intent_metrics['weighted_f1']}")
        print(f"  Escalation acc:   {esc_metrics['accuracy']}")
        print(f"  Escalation F1:    {esc_metrics['f1']}")
        print(f"  Reply quality:    {quality['mean_composite']}")
        print()

        if name != "majority_baseline":
            print("  Per-intent F1:")
            for intent, m in sorted(intent_metrics["per_intent"].items()):
                bar = "#" * int(m["f1"] * 20)
                print(f"    {intent:28s}  {m['f1']:.3f}  {bar}")
            print()

    # Judge-human agreement (on the agent's predictions)
    agent_preds = [agent.predict(row["text"]) for row in golden]
    agreement = judge_human_agreement(golden, agent_preds)
    results["judge_human_agreement"] = agreement

    print(f"{'='*60}")
    print("Judge-human agreement (Signal Desk agent)")
    print(f"{'='*60}")
    print(f"  Mean score (escalated):    {agreement['mean_score_escalated']}")
    print(f"  Mean score (auto-handled): {agreement['mean_score_autohandled']}")
    print(f"  Gap (auto - esc):          {agreement['score_gap']}")
    print()

    # Failure analysis
    failures = failure_analysis(golden, agent_preds)
    results["failure_analysis"] = failures

    print(f"{'='*60}")
    print("Failure analysis (Signal Desk agent)")
    print(f"{'='*60}")
    print(f"  Total failures: {failures['total_failures']} / {failures['total_samples']} ({failures['failure_rate']*100:.1f}%)")
    print()
    for i, mode in enumerate(failures["top_5_modes"], 1):
        print(f"  {i}. {mode['mode']} (n={mode['count']})")
        print(f"     Example: \"{mode['example'][:80]}...\"")
        print(f"     True: {mode['example_true']} | Predicted: {mode['example_pred']}")
        print()

    # Save full results
    output_path = ROOT / "data" / "eval_results.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Full results saved to {output_path}")

    if output_json:
        print("\n" + json.dumps(results, indent=2, default=str))

    return results


if __name__ == "__main__":
    run_evaluation(output_json="--json" in sys.argv)
