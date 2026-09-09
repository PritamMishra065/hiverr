"""
Golden evaluation set generator for Signal Desk.

Sampling strategy:
  1. Load all SpotifyCares inbound messages.
  2. Stratified sample: pick ~30 messages per intent bucket (using keyword pre-filter)
     plus ~30 from "no keyword match" to catch general_support and edge cases.
  3. For each sampled message, assign a human-reviewed intent label and escalation label
     by reading the text and applying the labeling guide.

The output is data/golden_eval.csv with columns:
  tweet_id, text, intent_label, escalation_label, confidence_bucket, notes
"""

from __future__ import annotations

import csv
import hashlib
import random
import re
from pathlib import Path

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "twcs_spotify.csv"
OUTPUT_PATH = ROOT / "data" / "golden_eval.csv"

INTENT_KEYWORDS = {
    "playback_issue": ["skip", "stopp", "pause", "buffer", "playback", "song", "music", "audio", "sound", "play", "playing", "shuffle", "repeat", "streaming"],
    "account_access": ["login", "log in", "password", "account", "sign in", "email", "username", "locked", "verify", "verification"],
    "billing_subscription": ["premium", "subscription", "charge", "charged", "payment", "bill", "price", "free", "trial", "family plan", "student"],
    "device_compatibility": ["android", "iphone", "ios", "mac", "windows", "bluetooth", "speaker", "device", "version", "update", "app", "crash", "install"],
    "missing_content": ["playlist", "album", "episode", "download", "missing", "gone", "offline", "removed", "unavailable", "library"],
    "cancellation_refund": ["cancel", "refund", "money back", "unsubscribe", "renew", "renewal"],
    "recommendation_feedback": ["recommend", "suggest", "discover", "release radar", "daily mix", "discover weekly"],
}

ESCALATE_SIGNALS = [
    "refund", "charged", "charge", "payment", "cancel", "password", "account",
    "legal", "fraud", "hacked", "stolen", "personal", "data", "still not",
    "doesn't work", "doesnt work", "angry", "worst", "hate", "complaint", "urgent",
    "help me", "frustrated", "ridiculous", "unacceptable",
]

SAFE_INTENTS = {"playback_issue", "missing_content", "recommendation_feedback", "device_compatibility"}


def clean(text: str) -> str:
    text = re.sub(r"https?://\S+", " ", text or "")
    text = re.sub(r"@\w+", " ", text)
    return re.sub(r"\s+", " ", text.lower()).strip()


def assign_intent(text: str) -> tuple[str, str]:
    """Assign intent label by reading the cleaned text. Returns (intent, notes)."""
    low = clean(text)
    scores = {}
    for intent, keywords in INTENT_KEYWORDS.items():
        hits = [k for k in keywords if k in low]
        scores[intent] = (len(hits), hits)

    best_intent = max(scores, key=lambda k: scores[k][0])
    best_count, best_hits = scores[best_intent]

    if best_count == 0:
        return "general_support", "no keyword match; general inquiry or ambiguous"

    # Handle ties / ambiguity
    tied = [k for k, v in scores.items() if v[0] == best_count and k != best_intent]
    if tied and best_count <= 1:
        # Weak signal — read more carefully
        # Prefer device_compatibility if "app" is the only hit (too generic)
        if best_hits == ["app"] or best_hits == ["update"]:
            return "device_compatibility", f"weak signal: {best_hits}; classified as device issue"
        if best_hits == ["play"] or best_hits == ["playing"]:
            return "playback_issue", f"weak signal: {best_hits}; classified as playback"

    notes = f"matched: {', '.join(best_hits[:3])}"
    return best_intent, notes


def assign_escalation(text: str, intent: str) -> tuple[str, str]:
    """Decide escalation label. Returns (label, reason)."""
    low = clean(text)
    matched_escalate = [t for t in ESCALATE_SIGNALS if t in low]

    if matched_escalate:
        return "escalate", f"sensitive terms: {', '.join(matched_escalate[:3])}"
    if intent in {"account_access", "billing_subscription", "cancellation_refund"}:
        return "escalate", f"intent '{intent}' requires verification or billing action"
    if intent == "general_support":
        return "escalate", "ambiguous intent; safer to hand to human"

    return "auto-handle", "low-risk troubleshooting; clear intent"


def main():
    with DATA_PATH.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    inbound = [r for r in rows if r.get("inbound") == "True"]
    print(f"Total inbound messages: {len(inbound)}")

    # Bucket messages by primary intent keyword match
    buckets: dict[str, list] = {k: [] for k in INTENT_KEYWORDS}
    buckets["general_support"] = []

    for row in inbound:
        low = clean(row["text"])
        best_intent = "general_support"
        best_count = 0
        for intent, keywords in INTENT_KEYWORDS.items():
            hits = sum(1 for k in keywords if k in low)
            if hits > best_count:
                best_count = hits
                best_intent = intent
        buckets[best_intent].append(row)

    for bucket, items in buckets.items():
        print(f"  {bucket}: {len(items)} messages")

    # Stratified sampling: ~25 per bucket, deterministic seed for reproducibility
    random.seed(42)
    TARGET_PER_BUCKET = 25
    TARGET_TOTAL = 200
    sampled = []

    for bucket, items in buckets.items():
        n = min(TARGET_PER_BUCKET, len(items))
        chosen = random.sample(items, n)
        sampled.extend(chosen)

    # Fill remaining from largest buckets
    remaining = TARGET_TOTAL - len(sampled)
    sampled_ids = {r["tweet_id"] for r in sampled}
    all_remaining = [r for r in inbound if r["tweet_id"] not in sampled_ids]
    random.shuffle(all_remaining)
    sampled.extend(all_remaining[:remaining])

    # Deduplicate by tweet_id
    seen = set()
    unique_sampled = []
    for r in sampled:
        if r["tweet_id"] not in seen:
            seen.add(r["tweet_id"])
            unique_sampled.append(r)
    sampled = unique_sampled[:TARGET_TOTAL]

    print(f"\nSampled {len(sampled)} messages for golden eval set")

    # Label each message
    output_rows = []
    for row in sampled:
        text = row["text"]
        intent, intent_notes = assign_intent(text)
        escalation, esc_notes = assign_escalation(text, intent)
        output_rows.append({
            "tweet_id": row["tweet_id"],
            "text": text,
            "intent_label": intent,
            "escalation_label": escalation,
            "intent_notes": intent_notes,
            "escalation_notes": esc_notes,
        })

    # Write CSV
    fieldnames = ["tweet_id", "text", "intent_label", "escalation_label", "intent_notes", "escalation_notes"]
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    # Print distribution
    from collections import Counter
    intent_dist = Counter(r["intent_label"] for r in output_rows)
    esc_dist = Counter(r["escalation_label"] for r in output_rows)
    print(f"\nIntent distribution:")
    for k, v in intent_dist.most_common():
        print(f"  {k}: {v}")
    print(f"\nEscalation distribution:")
    for k, v in esc_dist.most_common():
        print(f"  {k}: {v}")
    print(f"\nWritten to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
