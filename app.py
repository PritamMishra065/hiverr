from __future__ import annotations

import csv
import json
import os
import re
from collections import Counter
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).parent
DATA_PATH = Path(os.getenv("TWCS_DATA", ROOT / "data" / "twcs_spotify.csv"))
BRAND = "SpotifyCares"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")

# ── Groq client (lazy, only when key is set) ────────────────────────────────

_groq_client = None


def get_groq():
    global _groq_client
    if _groq_client is None and GROQ_API_KEY:
        from groq import Groq
        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client


def llm_available() -> bool:
    return bool(GROQ_API_KEY)


# ── Intent rules (fallback) ─────────────────────────────────────────────────

INTENT_RULES = {
    "playback_issue": ["skip", "stopp", "pause", "buffer", "playback", "song", "music", "audio", "sound"],
    "account_access": ["login", "log in", "password", "account", "sign in", "email", "username"],
    "billing_subscription": ["premium", "subscription", "charge", "charged", "payment", "bill", "price"],
    "device_compatibility": ["android", "iphone", "ios", "mac", "windows", "bluetooth", "speaker", "device", "version"],
    "missing_content": ["playlist", "album", "episode", "download", "missing", "gone", "offline"],
    "cancellation_refund": ["cancel", "refund", "money back", "unsubscribe", "renew"],
    "recommendation_feedback": ["recommend", "suggest", "discover", "release radar", "daily mix"],
}

INTENT_LIST = list(INTENT_RULES.keys()) + ["general_support"]

ESCALATE_TERMS = [
    "refund", "charged", "charge", "payment", "cancel", "password", "account", "legal", "fraud",
    "hacked", "stolen", "personal", "data", "still not", "doesn't work", "doesnt work", "angry",
    "worst", "hate", "complaint", "urgent",
]


def clean(text: str) -> str:
    text = re.sub(r"https?://\S+", " ", text or "")
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9'!? ]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


# ── Data loading ─────────────────────────────────────────────────────────────

def load_rows():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    with DATA_PATH.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["clean_text"] = clean(row.get("text", ""))
    return rows


ROWS = load_rows()
INBOUND = [r for r in ROWS if r.get("inbound") == "True"]
OUTBOUND = [r for r in ROWS if r.get("inbound") == "False" and r.get("author_id") == BRAND]
OUTBOUND_TEXT = [r["clean_text"] for r in OUTBOUND]
VECTOR = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30000, sublinear_tf=True)
OUTBOUND_MATRIX = VECTOR.fit_transform(OUTBOUND_TEXT) if OUTBOUND_TEXT else None


# ── Keyword-based classify (fallback) ────────────────────────────────────────

def classify_keywords(text: str):
    value = clean(text)
    scores = {intent: sum(1 for term in terms if term in value) for intent, terms in INTENT_RULES.items()}
    best = max(scores, key=scores.get) if scores and max(scores.values()) else "general_support"
    matched = scores.get(best, 0)
    confidence = min(0.96, 0.42 + matched * 0.13)
    return best, round(confidence, 2), [term for term in INTENT_RULES.get(best, []) if term in value][:4]


# ── LLM-powered classify (Groq) ─────────────────────────────────────────────

def classify_llm(text: str):
    client = get_groq()
    if not client:
        return classify_keywords(text)
    prompt = (
        f"You are an intent classifier for SpotifyCares customer support.\n"
        f"Classify the following customer message into exactly ONE of these intents:\n"
        f"{', '.join(INTENT_LIST)}\n\n"
        f"Customer message: \"{text}\"\n\n"
        f"Respond with ONLY a JSON object: {{\"intent\": \"<intent>\", \"confidence\": <0.0-1.0>, \"cues\": [\"keyword1\", \"keyword2\"]}}"
    )
    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=150,
        )
        raw = resp.choices[0].message.content.strip()
        # Extract JSON from response
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
            intent = data.get("intent", "general_support")
            if intent not in INTENT_LIST:
                intent = "general_support"
            confidence = max(0.0, min(1.0, float(data.get("confidence", 0.7))))
            cues = data.get("cues", [])[:4]
            return intent, round(confidence, 2), cues
    except Exception:
        pass
    return classify_keywords(text)


def classify(text: str):
    if llm_available():
        return classify_llm(text)
    return classify_keywords(text)


# ── Retrieval ────────────────────────────────────────────────────────────────

def retrieve(text: str, limit: int = 3):
    if not OUTBOUND or OUTBOUND_MATRIX is None:
        return []
    query = VECTOR.transform([clean(text)])
    scores = cosine_similarity(query, OUTBOUND_MATRIX).ravel()
    indices = scores.argsort()[::-1]
    results = []
    seen = set()
    for index in indices:
        row = OUTBOUND[int(index)]
        fingerprint = clean(row["text"])
        if fingerprint in seen or scores[index] < 0.08:
            continue
        seen.add(fingerprint)
        results.append({
            "text": row["text"],
            "score": round(float(scores[index]), 2),
            "date": row.get("created_at", ""),
            "tweet_id": row.get("tweet_id", ""),
        })
        if len(results) >= limit:
            break
    return results


# ── Escalation decision ─────────────────────────────────────────────────────

def decide(text: str, intent: str, confidence: float, evidence):
    value = clean(text)
    matched = [term for term in ESCALATE_TERMS if term in value]
    if matched:
        return "escalate", "Sensitive or high-risk language detected: " + ", ".join(matched[:3]) + "."
    if confidence < 0.55:
        return "escalate", "Intent confidence is below the safe auto-handling threshold."
    if not evidence or evidence[0]["score"] < 0.16:
        return "escalate", "No sufficiently similar historical resolution was retrieved."
    if intent in {"account_access", "billing_subscription", "cancellation_refund"}:
        return "escalate", "This intent may require account verification or a billing action."
    return "auto-handle", "Low-risk troubleshooting request with a strong historical match."


# ── Template-based draft reply (fallback) ────────────────────────────────────

def draft_reply_template(text: str, intent: str, decision: str, evidence):
    if decision == "escalate":
        return (
            "Thanks for reaching out — we want to get this resolved properly. "
            "Please send us a DM with the details of your account and the steps "
            "you have already tried so our support team can investigate securely."
        )
    if not evidence:
        return (
            "Thanks for reaching out! Could you share a little more about what "
            "you are seeing, including your device and the steps you have tried? "
            "We will help from there."
        )
    precedent = clean(evidence[0]["text"])
    actions = []
    if any(w in precedent for w in ["log out", "logging out", "log in"]):
        actions.append("logging out and back in")
    if any(w in precedent for w in ["restart", "restarting"]):
        actions.append("restarting your device")
    if any(w in precedent for w in ["reinstall", "reinstalling", "uninstall"]):
        actions.append("reinstalling the app")
    if any(w in precedent for w in ["dm", "direct message"]):
        actions.append("sending us a DM with your account email")
    if any(w in precedent for w in ["device", "version", "operating"]):
        actions.append("sharing your device and app version")
    if any(w in precedent for w in ["update", "latest"]):
        actions.append("making sure the app is up to date")
    if any(w in precedent for w in ["cache", "clear"]):
        actions.append("clearing the app cache")
    if any(w in precedent for w in ["shuffle", "repeat"]):
        actions.append("toggling shuffle/repeat off and on")
    if actions:
        steps = ", ".join(actions[:3])
        return (
            f"Thanks for reaching out! Based on how we have resolved similar issues, "
            f"could you try {steps}? If the issue continues, reply back with more "
            f"details and we will dig deeper."
        )
    return (
        "Thanks for reaching out! Could you share a little more about what "
        "you are seeing, including your device and the steps you have tried? "
        "We will help from there."
    )


# ── LLM-powered draft reply (Groq) ──────────────────────────────────────────

def draft_reply_llm(text: str, intent: str, decision: str, evidence):
    if decision == "escalate":
        return (
            "Thanks for reaching out — we want to get this resolved properly. "
            "Please send us a DM with the details of your account and the steps "
            "you have already tried so our support team can investigate securely."
        )
    client = get_groq()
    if not client:
        return draft_reply_template(text, intent, decision, evidence)

    evidence_text = "\n".join(
        f"- (score {e['score']}): {e['text']}" for e in evidence[:3]
    ) if evidence else "No similar historical replies found."

    prompt = (
        f"You are a SpotifyCares support agent. Draft a short, helpful reply to this customer message.\n\n"
        f"Customer message: \"{text}\"\n"
        f"Detected intent: {intent}\n"
        f"Decision: {decision}\n\n"
        f"Here are similar historical SpotifyCares replies for reference — stay grounded in this style:\n"
        f"{evidence_text}\n\n"
        f"Rules:\n"
        f"- Keep the reply under 280 characters (Twitter limit).\n"
        f"- Be empathetic, professional, and match SpotifyCares tone.\n"
        f"- Suggest concrete troubleshooting steps grounded in the historical replies.\n"
        f"- Do NOT make promises about fixes or timelines.\n"
        f"- Do NOT ask for passwords or sensitive info in the reply.\n\n"
        f"Reply with ONLY the draft reply text, nothing else."
    )
    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200,
        )
        reply = resp.choices[0].message.content.strip().strip('"')
        if 20 < len(reply) < 350:
            return reply
    except Exception:
        pass
    return draft_reply_template(text, intent, decision, evidence)


def draft_reply(text: str, intent: str, decision: str, evidence):
    if llm_available():
        return draft_reply_llm(text, intent, decision, evidence)
    return draft_reply_template(text, intent, decision, evidence)


# ── Agent pipeline ───────────────────────────────────────────────────────────

def agent_result(text: str):
    intent, confidence, cues = classify(text)
    evidence = retrieve(text)
    decision, reason = decide(text, intent, confidence, evidence)
    return {
        "brand": BRAND,
        "message": text,
        "intent": intent,
        "confidence": confidence,
        "cues": cues,
        "decision": decision,
        "reason": reason,
        "reply": draft_reply(text, intent, decision, evidence),
        "evidence": evidence,
        "mode": "llm" if llm_available() else "keyword",
    }


def evaluation():
    golden_path = ROOT / "data" / "golden_eval.csv"
    if golden_path.exists():
        with golden_path.open(encoding="utf-8", newline="") as f:
            golden = list(csv.DictReader(f))
        correct_intent = 0
        correct_esc = 0
        confident = 0
        counts = Counter()
        for row in golden:
            intent, confidence, _ = classify_keywords(row["text"])
            counts[intent] += 1
            if intent == row.get("intent_label"):
                correct_intent += 1
            decision, _ = decide(row["text"], intent, confidence, retrieve(row["text"]))
            if decision == row.get("escalation_label"):
                correct_esc += 1
            confident += confidence >= 0.55
        n = len(golden)
        return {
            "sample_size": n,
            "source": "golden evaluation set (hand-labelled)",
            "intent_accuracy": round(correct_intent / max(1, n), 2),
            "escalation_accuracy": round(correct_esc / max(1, n), 2),
            "high_confidence_rate": round(confident / max(1, n), 2),
            "intent_distribution": counts,
            "retrieval_corpus": len(OUTBOUND),
            "mode": "llm" if llm_available() else "keyword",
        }
    samples = INBOUND[:: max(1, len(INBOUND) // 200)][:200]
    counts = Counter()
    confident = 0
    for row in samples:
        intent, confidence, _ = classify_keywords(row["text"])
        counts[intent] += 1
        confident += confidence >= 0.55
    return {
        "sample_size": len(samples),
        "source": "proxy audit (no golden set found)",
        "intent_distribution": counts,
        "high_confidence_rate": round(confident / max(1, len(samples)), 2),
        "retrieval_corpus": len(OUTBOUND),
        "mode": "llm" if llm_available() else "keyword",
    }


# ── Flask app ────────────────────────────────────────────────────────────────

app = Flask(__name__)


@app.get("/")
def index():
    return render_template(
        "index.html",
        brand=BRAND,
        inbound_count=len(INBOUND),
        outbound_count=len(OUTBOUND),
        llm_mode=llm_available(),
    )


@app.post("/api/agent")
def run_agent():
    payload = request.get_json(silent=True) or {}
    text = (payload.get("message") or "").strip()
    if not text:
        return jsonify({"error": "Please enter a customer message."}), 400
    return jsonify(agent_result(text))


@app.get("/api/evaluation")
def get_evaluation():
    result = evaluation()
    result["intent_distribution"] = dict(result["intent_distribution"])
    return jsonify(result)


@app.get("/api/examples")
def examples():
    examples = []
    for row in INBOUND:
        if len(examples) >= 6:
            break
        if len(row["clean_text"]) > 24:
            examples.append({"text": row["text"], "id": row["tweet_id"]})
    return jsonify(examples)


@app.get("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "brand": BRAND,
        "inbound": len(INBOUND),
        "outbound": len(OUTBOUND),
        "mode": "llm" if llm_available() else "keyword",
    })


if __name__ == "__main__":
    mode = "LLM (Groq)" if llm_available() else "keyword rules (no GROQ_API_KEY)"
    print(f"Signal Desk starting in {mode} mode")
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "5000")),
        debug=os.getenv("FLASK_DEBUG", "false").lower() == "true",
    )
