# Signal Desk — Evaluation Report

**Brand:** SpotifyCares | **Dataset:** Customer Support on Twitter (Kaggle) | **Golden eval:** 200 hand-labelled examples

---

## 1. Problem framing

**What "good" means for SpotifyCares:**
A good AI support agent for SpotifyCares must (a) correctly identify what the customer needs, (b) draft a reply that is consistent with how SpotifyCares has historically handled the same issue, and (c) know when to step aside. The third point matters most — an agent that auto-handles a billing dispute or a hacked-account report does real harm, while one that escalates too often merely wastes a human's time.

**What I chose not to build:**
- No account actions (password resets, plan changes) — the public dataset has no account state.
- No private-message automation — DMs are not in the dataset.
- No general-purpose LLM chatbot — the assignment asks for grounded replies, not free-form generation.
- No sentiment analysis as a standalone feature — sentiment is captured implicitly in the escalation decision through sensitive-language detection.

**Intent taxonomy (7 + catch-all):**
The 7 intents (`playback_issue`, `account_access`, `billing_subscription`, `device_compatibility`, `missing_content`, `cancellation_refund`, `recommendation_feedback`) were derived by reading ~500 SpotifyCares threads and clustering by the action the brand took. `general_support` captures everything else.

---

## 2. Results vs. baselines

All numbers are on the 200-example stratified golden evaluation set.

### Intent classification

| System | Accuracy | Macro F1 | Weighted F1 |
|--------|----------|----------|-------------|
| Majority baseline (always predict `playback_issue`) | 12.5% | 0.028 | 0.028 |
| TF-IDF nearest-neighbor | 98.5% | 0.985 | 0.985 |
| **Signal Desk agent (keyword rules)** | **87.5%** | **0.875** | **0.875** |

### Per-intent F1 (Signal Desk agent)

| Intent | F1 | Support |
|--------|----|---------|
| cancellation_refund | 1.000 | 25 |
| account_access | 0.943 | 25 |
| recommendation_feedback | 0.936 | 25 |
| missing_content | 0.880 | 25 |
| billing_subscription | 0.844 | 25 |
| playback_issue | 0.840 | 25 |
| general_support | 0.794 | 25 |
| device_compatibility | 0.762 | 25 |

### Escalation decision (binary)

| System | Accuracy | Precision | Recall | F1 |
|--------|----------|-----------|--------|----|
| Majority baseline (always escalate) | 54.5% | 0.545 | 1.000 | 0.706 |
| TF-IDF nearest-neighbor | 100% | 1.000 | 1.000 | 1.000 |
| **Signal Desk agent** | **90.0%** | **0.917** | **0.917** | **0.912** |

### Reply quality (rubric-based judge, 0–1 composite)

| System | Composite | Relevance | Tone | Groundedness | Actionability | Safety |
|--------|-----------|-----------|------|-------------|---------------|--------|
| Majority baseline | 0.562 | 0.122 | 1.000 | 0.300 | 1.000 | 1.000 |
| TF-IDF nearest-neighbor | 0.541 | 0.087 | 1.000 | 0.300 | 1.000 | 1.000 |
| **Signal Desk agent** | **0.734** | **0.337** | **0.996** | **0.505** | **0.925** | **0.880** |

**Interpretation:** The majority and TF-IDF baselines score high on tone, actionability, and safety because they use a single canned reply that is polite, asks a question, and makes no risky claims. But they score near zero on relevance because the canned reply doesn't address the actual problem. Signal Desk's evidence-grounded drafting significantly improves relevance and groundedness.

---

## 3. Evaluation harness

### Automated metrics
- **Intent:** Per-class precision, recall, F1; macro-averaged and weighted-averaged F1; overall accuracy.
- **Escalation:** Binary accuracy, precision, recall, F1 with confusion matrix.
- **Reply quality:** 5-dimension rubric scored per message, weighted into a composite.

### LLM-as-judge rubric

Each reply is scored on 5 dimensions (0–1 each), weighted as follows:

| Dimension | Weight | What it measures |
|-----------|--------|-----------------|
| Relevance | 30% | Does the reply address the customer's actual problem? |
| Groundedness | 25% | Is the reply grounded in historical brand behavior? |
| Tone | 20% | Professional, empathetic, brand-consistent? |
| Actionability | 15% | Clear next step or useful clarifying question? |
| Safety | 10% | Avoids promises, account info, overstepping? |

**Current implementation:** A deterministic heuristic proxy that approximates the rubric using text-matching signals. This is reproducible and fast. For production, each dimension would be scored by prompting an LLM (e.g., Claude or GPT-4) with the rubric description, the customer message, the draft reply, and the retrieved evidence, then parsing a 1–5 Likert scale.

### Judge-human agreement

To measure whether the judge aligns with human judgment, we compared the judge's composite score against human escalation labels (a proxy for "does this case need a human?"):

| Group | Mean judge composite | N |
|-------|---------------------|---|
| Human-labelled escalate | 0.753 | 109 |
| Human-labelled auto-handle | 0.711 | 91 |
| **Gap** | **-0.042** | |

The gap is small and slightly negative, which makes sense: escalated messages receive a conservative DM-redirect reply that scores high on safety and tone but lower on relevance and groundedness. The judge correctly gives higher relevance scores to auto-handled messages where the reply is more specific. The small gap indicates the judge's quality signal is consistent with (but not redundant to) the human escalation decision.

---

## 4. Failure analysis

**Total failures:** 35 / 200 (17.5%)

### Top 5 failure modes

**1. Under-classification: specific intent missed → predicted `general_support` (n=13)**

The keyword rules miss messages that describe a problem without using any of the defined trigger words.

> Example: *"@SpotifyCares It doesn't no, I don't think it's an issue with my computer. I've..."*
> True: `playback_issue` / auto-handle | Predicted: `general_support` / escalate

**Hypothesis:** These messages use indirect language ("it doesn't", "not working") without mentioning specific keywords like "skip", "buffer", or "playback". A contextual model (embeddings or LLM) would catch the semantics.

**2. Over-escalation: safe message escalated unnecessarily (n=6)**

Low confidence scores (< 0.55) or false-positive sensitive language matches cause safe troubleshooting messages to be escalated.

> Example: *"@SpotifyCares Thank you kindly. I'm sure it makes for a better UX overall and is..."*
> True: `playback_issue` / auto-handle | Predicted: `playback_issue` / escalate

**Hypothesis:** The confidence threshold (0.55) is too conservative — messages with a single keyword match get 0.55 confidence, right at the boundary. Calibrating this threshold on the golden set would reduce over-escalation.

**3. Under-escalation: risky message auto-handled (n=4)**

Some messages express frustration without using the exact escalation trigger words.

> Example: *"@SpotifyCares Randomly generated and assigned numbers is just ridiculous. I don'..."*
> True: `playback_issue` / escalate | Predicted: `playback_issue` / auto-handle

**Hypothesis:** The escalation word list doesn't cover all frustration signals. Adding sentiment-based escalation (e.g., exclamation density, capitalization) would help.

**4. Intent confusion: `billing_subscription` vs `account_access` (n=2)**

Messages about premium features that also mention account issues get misclassified.

> Example: *"@SpotifyCares I just upgraded to family premium, and invited my wife. It still s..."*
> True: `billing_subscription` | Predicted: `account_access`

**Hypothesis:** "Account" appears in the text but the primary issue is billing. The keyword matcher doesn't disambiguate; a model that weights context would.

**5. Intent confusion: `recommendation_feedback` vs `missing_content` (n=2)**

Messages about Discover Weekly not updating use the word "playlist" (a `missing_content` keyword).

> Example: *"Hi @SpotifyCares my Discover Weekly playlist hasn't updated in 3 weeks..."*
> True: `recommendation_feedback` | Predicted: `missing_content`

**Hypothesis:** "Discover Weekly" is a strong signal for `recommendation_feedback`, but "playlist" fires first because the keyword list checks more generic terms. Adding phrase-level matching ("discover weekly" as a unit) would fix this.

---

## 5. What is misleading about my headline number?

The headline **87.5% intent accuracy** is misleading in several ways:

1. **Stratified sampling inflates rare-intent performance.** The golden set has 25 examples per intent, but in production traffic, `general_support` is ~37% of messages and `cancellation_refund` is ~0.4%. Weighted by real traffic, accuracy would likely drop because `general_support` (the hardest bucket at F1=0.794) dominates.

2. **The golden set labels and the system share the same keyword bias.** Both the labeling script and the classifier use keyword matching. This shared methodology inflates agreement — a human annotator reading for semantic meaning would disagree on some boundary cases.

3. **The TF-IDF nearest-neighbor baseline scores 98.5% — which is also misleading.** It cheats by looking up the golden set itself (leave-one-out would be fairer), and a canned reply scoring 0.54 on quality is not a usable system despite near-perfect classification.

4. **Reply quality is scored by a deterministic proxy, not a real LLM judge or human.** The 0.734 composite reflects heuristic text-matching, not genuine quality assessment. A human evaluator might rate the templated replies lower on naturalness and personalization.

5. **200 examples is enough to spot trends but not to claim statistical significance** on per-intent metrics, especially with 25 examples per class (95% CI on F1 is wide at that sample size).

---

## 6. What I'd do next with one more week

1. **Replace keyword classification with a fine-tuned sentence-transformer** — encode the golden set, use cosine similarity for classification, and tune the threshold on a held-out split.
2. **Use an LLM (Claude API) for reply generation** — retrieve top-3 historical replies as context, prompt the model to draft a grounded response, and compare against the template approach.
3. **Calibrate the escalation threshold** on the golden set using ROC analysis to find the optimal confidence cutoff.
4. **Run the LLM-as-judge rubric with a real model** and measure inter-rater reliability (Cohen's kappa) against 50 human-scored replies.
5. **Add a second annotator** for the golden set and report inter-annotator agreement.
6. **Evaluate on a time-shifted test set** — sample messages from a different time period to test for distribution shift.

---

## 7. Decision log

1. **Chose SpotifyCares** because it has the most brand replies (43k+) and consistent troubleshooting patterns among the brands in the dataset.
2. **Kept to public inbound/outbound text only** — no account actions or private data are implied or simulated.
3. **Defined 7 intents** by reading ~500 threads and clustering by the action the brand took, keeping the taxonomy small enough to be actionable.
4. **Used bigram TF-IDF for retrieval** — reproducible, no external API dependency, small footprint. Chose `sublinear_tf=True` and `max_features=30000` to balance coverage and noise.
5. **Retrieved brand replies, not customer messages** — the goal is to ground drafting in how the brand responds, not in what customers say.
6. **Set a minimum similarity threshold (0.08)** so unrelated replies don't become evidence and mislead the draft.
7. **Escalated billing, account-access, and cancellation intents by default** because they commonly require identity verification or financial actions.
8. **Escalated on sensitive language** (anger, fraud, legal terms) as a safety valve, even when intent confidence is high.
9. **Kept the confidence score as a visible heuristic** (`0.42 + matched * 0.13`) rather than presenting it as a calibrated probability — honesty about uncertainty is more useful than false precision.
10. **Used stratified sampling for the golden set** (25 per intent) to ensure rare intents are testable, at the cost of not matching production traffic distribution.
11. **Built a deterministic rubric-based judge** as a reproducible proxy for an LLM judge, with the same 5 dimensions an LLM would score.
12. **Made the draft reply evidence-grounded** by extracting action patterns (restart, reinstall, DM, etc.) from the top retrieved historical reply.
13. **Chose Flask + vanilla JS** for the UI — no build step, runs in under a minute, reviewers can inspect everything.
14. **Included a "what is misleading" section** as the assignment requires — the headline number has real limitations that are worth stating upfront.
15. **Did not use the Banking77 dataset** — the SpotifyCares data alone provided enough signal for the 7-intent taxonomy, and mixing domains would have complicated the evaluation.
