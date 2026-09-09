# Golden evaluation set — sampling and labeling guide

## Sampling strategy

The golden evaluation set contains **200 hand-labelled examples** drawn from
the SpotifyCares inbound messages in `twcs_spotify.csv`.

**Stratified sampling** was used to ensure representation across all 7 intent
categories plus the catch-all `general_support` bucket:

1. Each inbound message was pre-bucketed by keyword match into one of 8 groups.
2. 25 messages were sampled from each bucket (random seed = 42 for
   reproducibility).
3. This avoids the natural class imbalance where `general_support` and
   `playback_issue` dominate, and ensures rare intents like
   `cancellation_refund` (124 total messages) and `recommendation_feedback`
   (239 total messages) are well-represented.

**Why stratified, not random?** A pure random sample of 200 from 31,353
messages would contain ~0-1 `cancellation_refund` examples — not enough to
measure per-intent performance.

## Labeling guide

Each message was labelled on two dimensions:

### Intent label (8 classes)

| Intent | Definition | Examples |
|--------|-----------|----------|
| `playback_issue` | Playback stops, skips, buffers, shuffles wrong, audio issues | "songs keep skipping", "no sound" |
| `account_access` | Cannot log in, password reset, email change, verification | "can't sign in", "forgot password" |
| `billing_subscription` | Premium charges, plan changes, family plan, student discount | "charged twice", "upgrade to family" |
| `device_compatibility` | App crashes, device-specific bugs, version/update issues | "crashes on iPhone", "won't update" |
| `missing_content` | Playlists gone, albums missing, downloads unavailable | "playlist disappeared", "album removed" |
| `cancellation_refund` | Cancel subscription, request refund, stop renewal | "cancel premium", "want refund" |
| `recommendation_feedback` | Discover Weekly, Daily Mix, Release Radar, suggestions | "discover weekly same songs" |
| `general_support` | Does not fit above categories, vague, or multi-topic | "this is terrible", "need help" |

**Tie-breaking rule:** When a message matches multiple intents, label by the
primary action the customer needs (e.g., "I was charged for premium but can't
log in" → `billing_subscription` because the billing issue is the primary
complaint).

### Escalation label (binary)

| Label | When to apply |
|-------|--------------|
| `escalate` | Message involves billing/account actions, sensitive language (anger, fraud, legal), ambiguous intent, or requires identity verification |
| `auto-handle` | Clear troubleshooting request, low risk, intent is unambiguous, and similar issues have been resolved with standard steps |

## Quality notes

- Labels were assigned by reading each message's full text (not just keywords).
- The labeling was done in a single pass with the rules above, then spot-checked
  on a ~10% sub-sample for consistency.
- This is a first-pass golden set. For production use, a second annotator and
  inter-annotator agreement measurement would be needed.
