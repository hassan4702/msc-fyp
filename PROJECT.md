# Multimodal Emotion-Aware Chatbot — Complete Project Documentation

**Project:** Empath — a chatbot that reads emotion from a person's **face (webcam)** and
**typed text** at the same time, fuses the two signals, and uses the result to reply with
empathy.

**Context:** MSc final-year project — Abertay University, module **CMP504**, MSc **Applied AI
& UX**. Supervisor: Stuart Anderson.

**Repository:** https://github.com/hassan4702/msc-fyp

This document explains, end to end, **what** was built, **how**, **why**, and the **current
implementation** of every part.

---

## Table of contents
1. [The problem and motivation](#1-the-problem-and-motivation)
2. [Aim, objectives and research questions](#2-aim-objectives-and-research-questions)
3. [System architecture](#3-system-architecture)
4. [The seven-emotion label space](#4-the-seven-emotion-label-space)
5. [The models](#5-the-models)
6. [Datasets](#6-datasets)
7. [Fusion — the core contribution](#7-fusion--the-core-contribution)
8. [The response layer (LLM)](#8-the-response-layer-llm)
9. [Evaluation and results](#9-evaluation-and-results)
10. [Implementation — backend](#10-implementation--backend)
11. [Implementation — frontend](#11-implementation--frontend)
12. [Authentication and saved chats](#12-authentication-and-saved-chats)
13. [Key engineering challenges (and how they were solved)](#13-key-engineering-challenges-and-how-they-were-solved)
14. [Ethics and data protection](#14-ethics-and-data-protection)
15. [Testing](#15-testing)
16. [Project structure](#16-project-structure)
17. [How to run it](#17-how-to-run-it)
18. [Current status and future work](#18-current-status-and-future-work)
19. [Tech stack summary](#19-tech-stack-summary)

---

## 1. The problem and motivation

Most chatbots respond only to the **words** a person types. But people do not always say what
they feel — someone types *"I'm fine"* while looking visibly upset. A words-only system misses
that mismatch and replies in the wrong tone, which is worst exactly when tone matters most
(support, wellbeing, distress).

**The idea:** if a chatbot can *also see the user's face*, it has a second, independent clue to
their emotional state, and can respond more like an attentive human. This project builds that
system and asks one honest question: **does adding the face actually help — and how should the
bot behave when the face and the words disagree?**

**Why it matters / applications:** tutoring bots that notice frustration, customer-service bots
that catch genuine distress, wellbeing/mental-health assistants, and any assistant where
emotional tone changes what a good reply looks like.

---

## 2. Aim, objectives and research questions

**Aim:** design, build and evaluate a multimodal emotion-aware chatbot that fuses facial and
textual emotion to generate more empathetic, contextually appropriate responses.

**Research questions:**
- **RQ1** — Does combining face + text signals improve emotion-recognition accuracy over either
  channel alone?
- **RQ2** — How should conflicting signals be handled (face says happy, text says angry)?
- **RQ3** — Do users rate the multimodal chatbot's responses as more empathetic than a
  text-only baseline? *(planned user study — see §18.)*

**Original contribution:** not that multimodal emotion recognition is new, but taking fused
face+text emotion **all the way through to live response generation**, with an explicit,
evaluated **conflict-handling** policy.

---

## 3. System architecture

Every message flows through four stages:

```
[ webcam frame ] ─┐                         ┌─> face model  ─> P_face (7 probs) + confidence
                  ├─> [ FastAPI backend ] ──┤
[ typed text  ] ─┘                         └─> text model  ─> P_text (7 probs) + confidence
                                                    │
                                                    ▼
                                       FUSION (calibrated, gated,
                                        + conflict detection)
                                                    │
                                     fused emotion + "conflicted?" flag
                                                    │
                                                    ▼
                                    system prompt  ─> LLM  ─> empathetic reply
```

The whole system rests on one design decision: **both models output a probability distribution
over the *same* seven emotions.** That shared vocabulary is what makes late fusion possible at
all. The contracts live in `backend/models/base.py`.

---

## 4. The seven-emotion label space

The canonical labels are **Ekman's six basic emotions + neutral**: `anger, disgust, fear,
happy, sad, surprise, neutral` (`backend/emotions.py`).

**Why seven, not more?** The face is the binding constraint. Facial-emotion datasets (FER-2013)
label seven emotions, and a face cannot reliably show fine categories like "admiration" or
"gratitude." Since the project's whole point is *fusing* face and text, both channels must speak
the same vocabulary — so the shared set is capped at what both can express. Seven is also the
standard in cross-modal affective computing, which keeps the results comparable to prior work.

---

## 5. The models

Five models are involved. **Three were trained for this project**; two are used off-the-shelf.

| Role | Model | Trained here? | Data |
|------|-------|---------------|------|
| Text → emotion | DistilBERT (`distilbert-base-uncased`) | ✅ fine-tuned | GoEmotions |
| Face detection | MediaPipe BlazeFace (short-range) | ❌ off-the-shelf | — |
| Face → emotion | ResNet-18, ImageNet-pretrained | ✅ fine-tuned | FER-2013 |
| Fusion arbiter | Logistic Regression | ✅ trained | MELD |
| Reply generation | Qwen2.5-7B (Ollama) / Gemini 3.5 Flash | ❌ off-the-shelf | — |

An earlier from-scratch CNN (`FaceNet`) and an OpenCV Haar cascade filled the two face roles;
both were replaced, and §9 reports the before/after because the comparison is a result.

### 5.1 Text emotion model
- **What:** DistilBERT (a smaller, faster BERT) fine-tuned to classify a message into the 7 emotions.
- **How:** GoEmotions has 27 emotions; they are collapsed to 7 using the **official Ekman
  mapping** published by the GoEmotions authors (so the collapse is citable, not arbitrary).
  GoEmotions is multi-label; only examples whose labels fall into a single Ekman bucket are kept
  (~91% retained). Trained with a class-weighted loss (GoEmotions is imbalanced), early-stopped
  on validation macro-F1. Code: `training/train_text.py`, mapping in `training/goemotions.py`.
- **Result:** test **macro-F1 0.616**, accuracy 0.663 — in line with the published GoEmotions
  Ekman baseline (~0.64).
- **Why DistilBERT:** light enough to run on a weak deployment machine, close to BERT accuracy.

### 5.2 Face emotion model
- **What:** an ImageNet-pretrained **ResNet-18** (`backend/models/face_net.py:build_resnet18`),
  fine-tuned on FER-2013, over a 224×224 face crop → 7 emotions. It replaced a from-scratch
  0.98M-parameter CNN (`FaceNet`, still in the same file) which is retained as the baseline the
  results below are measured against.
- **How:** fine-tuned on FER-2013 (via the `Aaryan333/...` HF mirror: 28,709 train / 3,589
  validation / 3,589 test, the standard FER split), AdamW at 3e-4 with **cosine decay**, 15
  epochs, class-weighted loss with label smoothing, flip / rotation / brightness augmentation,
  best-on-validation checkpointing, seeded. Labels mapped **by name** (HF mirrors disagree on
  index order). Code: `training/train_face_resnet.py`, mapping in `training/fer2013.py`.
- **Crops come from the detector, at training time too.** Every image — train and inference
  alike — is framed by `backend/models/face_detect.py`. See §13 for why that is not a detail.

**Result on de-duplicated privateTest (3,301 images), against the from-scratch baseline:**

| | macro-F1 | accuracy |
|---|---|---|
| FaceNet, from scratch (previous) | 0.540 | 0.580 |
| **ResNet-18, fine-tuned (current)** | **0.651** | **0.691** |

Per-class F1: happy 0.90, surprise 0.78, neutral 0.68, anger 0.59, sad 0.56, **fear 0.55**,
disgust 0.50. Fear — the classic FER-2013 weak spot, and 0.32 under the old model — is the
single largest per-class gain. Accuracy of 0.691 is **above the ~65% human benchmark** on
FER-2013 and within a few points of the ~73% published state of the art.
- **Why "de-duplicated" — FER-2013 leaks.** 288 of the 3,589 privateTest images (**8.02%**) and
  280 of the validation images (7.80%) are pixel-identical to a training image, found by MD5 of
  the raw pixel buffer. The old model scored **0.781 accuracy on the leaked rows** against 0.580
  on the rest, so an undeduplicated figure is part memorisation — worth 1.6pp of accuracy and
  4.1pp of macro-F1 on its own. This is a defect of the FER-2013 distribution, not of the
  training code. `training/train_face_resnet.py` drops the leaked rows from validation and test
  before scoring, so **every FER number in this document is on clean data**.
- **Why a pretrained backbone.** Training 0.98M parameters from scratch on 28k 48×48 grayscale
  images caps out around the mid-50s; that was the binding limit, not the fusion design. The
  deployment-cost objection did not survive measurement — ResNet-18 runs at **7.7 ms/frame on 4
  CPU threads**, against ~23 ms/frame for the Haar cascade it replaced.
- **At runtime:** MediaPipe BlazeFace finds the face in the webcam frame, the box is expanded by
  a 25% margin, and the crop is resized to 224×224 and classified. The frame is decoded in
  **colour** for detection — BlazeFace loses 22.6 points of coverage on grayscale input (96.3% →
  73.7%, measured on 300 MELD frames) — and the crop handed to the classifier is grayscale. If no
  face is detected the channel reports "unavailable" and fusion falls back to text
  (`backend/models/face_model.py`).

### 5.3 Calibration
Neural nets are overconfident, and the two models' confidences must be **comparable** before
one can be trusted over the other. Each model's logits are **temperature-scaled** — a single
fitted number per model that softens overconfidence without changing the predicted label.
**A fitted temperature is now only adopted if it earns its place.** `training/calibrate.py` fits
T on half the validation split and then checks ECE on the other half; if scaling does not improve
held-out ECE, it keeps T = 1.0. This is not ceremony — it caught a real problem. The validation
split is also the split the best checkpoint is selected on, so the model is mildly overfit to it
and the NLL-optimal temperature overshoots:

| Model | fitted T | held-out ECE at T=1.0 | at fitted T | adopted |
|---|---|---|---|---|
| Text (DistilBERT) | 1.169 | 0.0964 | **0.0499** | ✅ T = 1.169 |
| Face (ResNet-18) | 1.268 | **0.0477** | 0.0513 | ❌ rejected → T = 1.0 |

The text model is genuinely overconfident and temperature scaling nearly halves its ECE. The face
model is already well calibrated, and the fitted temperature made it *worse* — on the full test
split, T=1.29 gave ECE 0.069 where T=1.0 gave 0.028. The previous version of this file claimed a
calibration benefit for the face channel; that claim was wrong, and the earlier shipped value was
fitted on raw FER thumbnails while inference ran Haar boxes, so it described a pipeline that never
actually ran. Values stored in `<model>/calibration.json`; the wrappers load them automatically.

---

## 6. Datasets

| Dataset | Used for | Notes |
|---------|----------|-------|
| **GoEmotions** | training the text model | ~58k Reddit comments, 27 labels → collapsed to 7 (official Ekman mapping) |
| **FER-2013** | training the face model | ~35,900 grayscale 48×48 face images, 7 emotions |
| **MELD** | evaluating fusion (RQ1/RQ2) | ~13k video utterances from *Friends*; the neutral multimodal test set |

- **MELD detail:** the accessible HF mirror (`BigfufuOuO/meld_raw`, 11.9 GB) is an ASR
  repackaging — videos + transcripts but **no emotion labels**. The labels were pulled from the
  authoritative declare-lab `*_sent_emo.csv` files and joined to the videos by
  (split, dialogue, utterance).
- **Not used:** AffectNet (needs an access application; treated as optional) and ISEAR (its
  label set — shame/guilt, no neutral/surprise — doesn't match the 7).

---

## 7. Fusion — the core contribution

Both channels output a 7-dim probability vector + a (calibrated) confidence. Fusion turns them
into one decision. Three strategies of increasing sophistication (`backend/models/fusion.py`):

1. **Weighted average** — `P_fused = w·P_text + (1−w)·P_face`. Simple baseline.
2. **Confidence gating** — weights scale with each channel's confidence; if a channel is
   unavailable (e.g. no face) it is dropped; a **conflict flag** is set when both channels are
   confident but disagree.
3. **Learned arbiter (`LearnedFusion`)** — a logistic-regression classifier trained on
   `[P_text, P_face, conf_text, conf_face, face_available]` that *decides* the label, especially
   in conflict cases. This is the RQ2 answer, trained on MELD.

**Why the arbiter matters:** simple averaging is *wrong* when the channels confidently disagree
(happy + angry averages to mush). A learned policy handles disagreement instead of blending it
away.

---

## 8. The response layer (LLM)

The fused emotion is injected into a language model's system prompt to shape the reply.

- **Backends (auto-selected at startup, `LLM_BACKEND=auto`):**
  1. **Ollama + Qwen2.5-7B** — local, free, private (used when running).
  2. **Google Gemini** (`gemini-3.5-flash`) — cloud fallback when Ollama isn't available.
  3. **Template responder** — offline canned replies, last resort (also used in tests).
  Only the **emotion label and text** ever reach the LLM — never webcam frames (`backend/models/llm.py`).
- **The persona:** the system prompt casts the model as a "reader" — a warm, emotionally-attuned
  companion that reflects what's underneath what the user says and gently checks the detected
  emotion rather than announcing it. The fused emotion is passed as a *private cue*.
- **Guardrails (strict scope):** the model must only hold the emotional conversation. It refuses
  code, trivia, maths, lookups, translation, and medical/legal/financial advice, and resists
  jailbreaks. Because prompt-only rules are unreliable on a 7B model, there is a **deterministic
  backstop**: any reply containing a code block is replaced with a gentle refusal.

---

## 9. Evaluation and results

Two tracks, matching the research questions. Metric: **macro-averaged F1** over the 7 classes
(handles class imbalance), reported with per-class breakdowns and against baselines.

Four pipelines are reported, because the differences between them are themselves results.
**(A)** the original Haar + from-scratch CNN with the crop-framing bug; **(B)** the same models
with the crop fix of §13; **(C)** BlazeFace + ResNet-18; **(D)** (C) additionally fine-tuned on
MELD's own train split (`training/finetune_meld.py`).

### RQ1 — MELD test set (2,610 utterances)

| System | (A) original | (B) crop fixed | (C) ResNet-18 | (D) + MELD-tuned |
|--------|--------------|----------------|---------------|------------------|
| text-only | 0.2678 | 0.2678 | 0.2678 | 0.2678 |
| face-only | 0.1318 | 0.1398 | 0.1210 | **0.1506** |
| majority-class | 0.0928 | 0.0928 | 0.0928 | 0.0928 |
| fused: weighted-average | 0.2662 | 0.2655 | 0.2505 | 0.2547 |
| fused: confidence-gated | 0.2618 | 0.2640 | 0.2451 | 0.2586 |
| fused: learned arbiter *(train)* | 0.2693 | **0.2751** | 0.2679 | 0.1954 † |
| fused: learned arbiter *(dev)* | — | — | 0.2323 | 0.2314 |
| *face-detection coverage* | *88.7%* | *88.7%* | *96.6%* | *96.6%* |

### RQ2 — conflict subset (where the two channels disagree)

| System | (A) original | (B) crop fixed | (C) ResNet-18 | (D) + MELD-tuned |
|--------|--------------|----------------|---------------|------------------|
| text-only | 0.2561 | 0.2567 | 0.2591 | 0.2691 |
| fused: learned arbiter *(train)* | 0.2630 | **0.2648** | 0.2574 | 0.1754 † |
| fused: weighted-average | 0.2551 | 0.2542 | 0.2425 | 0.2535 |
| fused: confidence-gated | 0.2491 | 0.2520 | 0.2359 | 0.2571 |
| face-only | 0.0930 | 0.1006 | 0.0930 | 0.1173 |
| *subset size* | *2,025* | *1,969* | *2,167* | *2,090* |

### † The arbiter cannot be trained on a split the face model was fine-tuned on

The (D) figures marked † are **invalid, and are reported only to document the trap**. The arbiter
trains on MELD *train* records; in (D) the face model was itself fine-tuned on MELD *train*, so
the arbiter learns from face predictions on utterances the face model had memorised (its training
loss fell to 0.89). It over-trusts the face channel and then collapses on test: 0.2679 → 0.1954.
Pipelines (A)–(C) are immune because those face models never saw MELD — which is precisely what
makes the bug easy to miss the moment in-domain fine-tuning is introduced.
`evaluation/evaluate_meld.py --arbiter-split dev` trains the arbiter on a split the face model
saw only for checkpoint selection.

That fix gives 0.2314, still below (C)'s 0.2679 — which raises a second question: is that the
leak, or simply that dev holds 9× less arbiter training data (1,109 vs 9,989 records)? Running
**(C) under the same dev arbiter** isolates it:

| arbiter training split | (C) ResNet-18 | (D) MELD-tuned |
|---|---|---|
| train — 9,989 records | 0.2679 | 0.1954 † leaked |
| dev — 1,109 records | 0.2323 | 0.2314 |

Under matched conditions the two are **identical within noise** (0.2323 vs 0.2314). The drop was
arbiter data starvation, not the fine-tuned face model. A separate lesson falls out of the same
table: the learned arbiter is strongly **data-hungry**, losing 3.5pp when its training set shrinks
9× — worth stating, since the arbiter is this project's RQ2 contribution.

### The finding that matters most for RQ1

**Improving the face channel by 25% relative (0.1210 → 0.1506 macro-F1) produced no measurable
improvement in fusion** (0.2323 → 0.2314 under a matched arbiter), and no pipeline's fusion beat
text-only at 0.2678. The face channel is too weak relative to text for its quality to change the
outcome at these levels. That is the honest answer to RQ1 on MELD, and it is a stronger claim than
any single pipeline could support, because it now holds across four of them — including one whose
face channel was trained on MELD itself.

**Text-only is identical to four decimals across all three runs** — the control confirming that
only the face path changed.

### The uncomfortable finding: in-domain gains did not transfer

Pipeline (C) is **+11.1pp macro-F1 on FER-2013** (0.540 → 0.651, §5.2) and detects 8pp more faces,
yet it is **worse on MELD** (face-only 0.140 → 0.121). That is not a bug, and it is not the
coverage change either — the obvious explanation, that Haar's 11% missed faces were scored as
"neutral" and MELD is 48% neutral, was tested and **refuted**: restricted to the 2,242 utterances
where *both* pipelines detect a face, so neither gets a free "neutral", the old model still leads
**0.1305 to 0.1128**.

The honest reading is that the larger pretrained model fits FER-2013's specific distribution
(posed, frontal, 48×48 grayscale stills) more tightly, and therefore **generalises less well** to
MELD's very different distribution (TV dialogue frames, motion blur, profiles, cinematic
lighting). Higher in-domain accuracy bought worse cross-corpus transfer.

**Pipeline (D) confirms that diagnosis by reversing it.** Fine-tuning the same ResNet-18 on MELD's
own train split lifts face-only from 0.1210 to **0.1506** — the best face channel of any pipeline
here, and above the from-scratch model it had been losing to. So the (C) regression was domain
shift, exactly as claimed, and not a defect in the backbone.

**Which model ships: (C), not (D).** (D) is better *on MELD* and is the right choice for anyone
reporting MELD numbers. But the product is a **webcam** — a large, frontal, well-lit face, much
closer to FER-2013's setting than to TV stills — and (D) buys its MELD gain by specialising away
from that. (D) also delivers **no fusion improvement whatsoever** (see above), so it would trade
deployment-domain accuracy for nothing the system actually uses. `resnet18_meld.pt` is therefore
kept as an evaluated variant, not promoted. That reasoning is a judgement about domains, not a
measurement: **RQ3's user study is the only part of this project that measures the face channel in
its actual deployment domain**, and it is what would settle the choice properly.

### Findings (stated honestly)
- **Naive fusion does not beat text-only** — adding the weak face channel *slightly degrades*
  the strong text channel. This holds across all three pipelines, so it is a property of the
  fusion design and the channel-strength imbalance, not of any one face model.
- **The learned arbiter is the only strategy that avoids degradation.** In (A) and (B) it exceeds
  text-only overall and wins on the conflict subset; in (C) it matches text-only where naive
  fusion falls 1.7–2.3pp below it. That is the RQ2 evidence: a *learned* conflict policy is what
  stops a weak second channel from doing harm.
- **The binding constraint is face-channel quality, not the fusion rule** — and even that has a
  ceiling. Face-only sits at 0.12–0.15 macro-F1 against a 0.093 majority baseline and a ~0.14
  chance level for 7 classes. Raising it 25% relative, via in-domain fine-tuning, changed fusion
  by 0.001. No fusion policy extracts much from a channel this close to chance, which is why
  "does the face help?" cannot be answered cleanly on MELD at all.
- **A learned arbiter must not be trained on data its input models were trained on.** Fine-tuning
  the face channel in-domain silently contaminated the arbiter's training set and cost 7pp on
  test (§9 †). The fix is a one-flag change; noticing it required suspecting a result that had
  improved on every other axis.
- **Cross-corpus vs in-domain:** (A)–(C) are **transfer** numbers — trained on GoEmotions /
  FER-2013, applied to MELD with no MELD fine-tuning — which is why absolute F1 sits near 0.27,
  well below in-domain MELD text SOTA. (D) closes that gap for the face channel specifically, and
  demonstrates the gap was real: face-only rises 0.1210 → 0.1506 from in-domain training alone.
- Full write-up: `evaluation/results/meld_results.md`; harness: `evaluation/evaluate_meld.py`.

---

## 10. Implementation — backend

- **Framework:** FastAPI (`backend/app.py`), served by uvicorn. Endpoints: `GET /health`,
  `POST /chat` (message + optional base64 frames + history → reply + all three emotion views +
  conflict flag), and `GET /` (serves the plain-HTML UI).
- **Pipeline** (`backend/services/pipeline.py`): orchestrates text + face → fusion → responder.
  It depends only on the abstract contracts, so stub models can be swapped for trained ones with
  no change to callers.
- **Config** (`backend/config.py`): environment-driven, loads a git-ignored root `.env`
  (model paths, `GEMINI_API_KEY`, backend choice). Stubs ship so the repo runs with no weights.
- **Robustness:** malformed webcam frames are caught and treated as "no face" rather than
  crashing the request.

---

## 11. Implementation — frontend

There are two frontends:

- **Plain HTML** (`frontend/index.html`) — a single self-contained page served directly by
  FastAPI at `localhost:8000`. Webcam + chat + a live emotion strip. No build step, no database.
  The simplest way to run the whole product.
- **Next.js app** (`web/`) — Next 16 + React 19 + Tailwind 4 + shadcn/ui (on Base UI). A
  custom warm "Empath" theme (Fraunces + Hanken Grotesk fonts) whose ambient accent **tints to
  the detected emotion**. Chat with markdown replies, per-message words/face/fused emotion chips,
  a sticky nav + composer, and (when signed in) a sidebar of saved chats. It calls the FastAPI
  backend through a **`/backend/*` rewrite proxy** (same-origin, no CORS), kept off `/api/*` so
  it never collides with the auth routes.

---

## 12. Authentication and saved chats

Built into the Next.js app (the Python backend stays a stateless emotion engine).

- **Auth:** Better Auth (email + password), self-hosted, free/MIT. Data stays in the project's
  own database.
- **Database:** Neon serverless Postgres, via Drizzle ORM. Tables: Better Auth's
  `user/session/account/verification` + the app's `conversation/message`.
- **Behaviour (as specified):**
  - **Anonymous** — chat works, nothing is saved, refresh clears the history.
  - **Signed in** — every turn is saved; a sidebar lists past chats to reopen; the account
    (email + log out) sits in the sidebar footer, while the sign-up button lives in the navbar
    when logged out.
- **Security:** the history API (`web/app/api/history`) is session-guarded and
  ownership-checked (a user can only read their own chats). Secrets live in git-ignored `.env`.

---

## 13. Key engineering challenges (and how they were solved)

These are real problems hit during the build — good evidence of engineering judgement:

- **No paired face+text dataset exists.** RQ1 needs both modalities on the same instance; the
  unimodal training sets don't provide that. Resolved by evaluating fusion on **MELD** (a
  multimodal set), accepting that its "text" is a speech transcript.
- **MELD had no emotion labels** in the accessible mirror. Solved by joining the videos to the
  authoritative declare-lab label CSVs by (split, dialogue, utterance).
- **Prompt-only guardrails failed** on the 7B model (it wrote code despite being told not to).
  Solved with a **deterministic backstop** (code detected → refusal), on top of a stronger prompt.
- **LLM portability** — the training machine is powerful but temporary; the deployment machine
  is weak. Solved by putting the LLM behind a swappable interface with **auto-fallback**
  (Ollama → Gemini → template).
- **Gemini model access** — `gemini-2.5-flash` is blocked for new API keys; switched the default
  to `gemini-3.5-flash`, which works.
- **Webcam not attaching (React)** — the `<video>` was conditionally rendered on a state that
  only flipped after the stream attached; fixed by always mounting it.
- **The face channel was classifying a different crop than it was trained on.** Training fed the
  model the full 48×48 FER-2013 thumbnail; inference fed it the raw Haar bounding box, which is
  ~0.807× that framing. Measured on 3,589 held-out FER images: accuracy 0.583 → 0.523 and
  macro-F1 0.566 → 0.491 from the framing alone. The first fix widened the Haar box by 1/8 per
  side; the structural fix was to move detection into `backend/models/face_detect.py` and have
  **the training script and the inference wrapper call the same function**, so the two framings
  cannot drift apart again. Nothing in the type system or the tests catches a disagreement in
  image geometry — only shared code does.
- **Detecting on grayscale threw away a fifth of the faces.** The pipeline decoded webcam frames
  with `IMREAD_GRAYSCALE` because the classifier wants grayscale — so the *detector* never saw
  colour either. BlazeFace is trained on colour and its coverage on the same 300 MELD frames is
  **96.3% on colour against 73.7% on grayscale**. Fixed by decoding in colour, detecting in
  colour, and converting to grayscale only for the classifier crop. The lesson: an optimisation
  made for one stage of a pipeline silently degraded an earlier stage.
- **Face alignment was investigated and rejected on evidence.** Warping the face onto canonical
  eye positions is standard FER preprocessing, so it was the obvious next step. MediaPipe's
  landmarker reaches 93% coverage on FER-2013 thumbnails but only **40.3% on MELD video frames**
  (48.3% even when run on a pre-detected crop). Aligning 93% of training images while being able
  to align under half of inference frames would have re-created the very train/inference mismatch
  described two bullets above. Measured first, not built — the cheapest engineering decision in
  this project.
- **A fitted calibration temperature made the model worse.** See §5.3: the face model's fitted
  T=1.29 raised held-out ECE from 0.028 to 0.069, because the temperature was being fitted on the
  same split the checkpoint was selected on. Fixed by making `calibrate.py` validate a fitted
  temperature on held-out data and fall back to T=1.0 when it does not help. The same guard
  *accepts* the text model's temperature, which nearly halves its ECE — so it discriminates
  rather than simply refusing to calibrate.
- **A silent stub was masquerading as the real model.** `FACE_MODEL_PATH` is relative, so the
  `os.path.isfile()` pre-check failed whenever the server was started from anywhere but the repo
  root — and because it failed *before* the `try`, the warning in the `except` never printed. The
  stub returns a constant `neutral / 0.60` with `available=True`, which is indistinguishable in
  the UI from a real prediction. Fixed by deleting the pre-check (so failures reach the logging
  path), resolving config paths against the repo root, and reporting the loaded class in
  `/health`. **Fail loudly, and make "which model am I actually running" observable.**
- **`except Exception: return None` hid every runtime failure as "no face."** Decode errors, MPS
  device errors, and a failed cascade load all surfaced identically to a user simply being out of
  frame. Fixed by logging the exception before returning.
- **The Haar detector is not thread-safe.** `/chat` is a sync endpoint, so FastAPI runs it in a
  threadpool against one shared `CascadeClassifier`. Under 180 concurrent calls this dropped ~4%
  of faces and returned corrupted boxes for ~5%; serial execution never did. Fixed with a lock
  around `detectMultiScale` (the torch forward measured clean and is left unlocked).

---

## 14. Ethics and data protection

- A **research risk assessment** was completed and supervisor-signed. Webcam data is biometric,
  special-category personal data under UK GDPR; the planned user study requires **ethics
  committee approval before any participant**.
- **Data handling:** raw video is processed in real time and **never persisted**; only the
  detected emotion label and the typed text reach the language model — never the images. The
  self-hosted auth + own database keep user data under the project's control.

---

## 15. Testing

- **51 unit tests** (`pytest`, in `tests/`): the label mappings, calibration, fusion strategies
  (including conflict detection), the pipeline, the API, and the guardrail backstop. Tests are
  hermetic (forced to the offline template responder + stub models via `conftest.py`).
- **End-to-end harness** (`scripts/e2e_check.py`): 24 cases against the live server — the seven
  emotions, text edge cases (empty, very long, unicode, prompt-injection, multiline), frame
  cases (no face, blank, malformed, real face), conversation history, and validation (missing
  field → 422). Runnable in a loop for stress testing.

---

## 16. Project structure

```
backend/        FastAPI app, model interfaces, fusion, responder, pipeline
  emotions.py       the 7-label space
  models/           base contracts, text/face models, fusion, face_net, llm
    face_detect.py    BlazeFace detection + canonical crop, SHARED with training
  services/         pipeline orchestration
  app.py            FastAPI: /health, /chat, / (HTML UI)
training/       fine-tuning + data-prep scripts (run on the powerful machine)
  train_face_resnet.py  the current face model
  finetune_meld.py      in-domain MELD fine-tuning
  train_face.py         the superseded from-scratch CNN (kept as the §9 baseline)
evaluation/     MELD harness, scoring, and saved results
frontend/       the plain-HTML UI
web/            the Next.js app (auth + saved chats)
tests/          pytest suite
scripts/        the end-to-end test harness
docs/plan/      the implementation plan
models/weights/ trained weights (git-ignored — sent separately)
```

---

## 17. How to run it

**Quickest (chatbot only, no database):**
```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-run.txt
ollama pull qwen3:4b            # optional; else it falls back to Gemini/template
LLM_BACKEND=auto uvicorn backend.app:app
# open http://localhost:8000
```
On **Windows**, double-click **`run.bat`** (it creates the environment and starts the server).

**Full web app (login + saved chats):** additionally run the Next.js app in `web/` with a Neon
database — see `SETUP.md`.

Requirements: Python 3.12; the trained weights in `models/weights/`; Ollama or a Gemini key for
real replies; Node 20+/pnpm + Neon only for the web app.

---

## 18. Current status and future work

**Done:** both emotion models trained + calibrated; three fusion strategies incl. the learned
arbiter; the MELD quantitative evaluation (RQ1 + RQ2); the empathetic response layer with the
reader persona, guardrails, and LLM auto-fallback; the plain-HTML UI; the Next.js web app with
auth and saved chats; 51 tests + an e2e harness.

**Remaining / future work, in priority order.** The honest constraint on this project is that the
**face channel is weak** (macro-F1 0.54 in-domain, ~0.13 on MELD against a 0.14 chance baseline).
That weakness is what makes RQ1 come out negative, so lifting it is the highest-value work — a
fusion experiment where one channel is near-chance cannot really answer "does the face help?"

**Done since the first draft of this section** (all measured, see §9 and §13): pretrained
ResNet-18 backbone (+11.1pp macro-F1 on FER-2013), MediaPipe BlazeFace replacing Haar (detection
24.7% → 100% on FER thumbnails, 52.7% → 98.7% at 30° head tilt, and 23× faster), colour decoding
for detection (+22.6pp coverage), the crop-framing fix (+6pp), LR scheduling, FER de-duplication,
and in-domain MELD fine-tuning. Two items were **investigated and rejected on evidence**: face
alignment (MediaPipe's landmarker reaches only 40.3% coverage on MELD frames, so aligning training
data would have re-created the very train/inference mismatch of §13) and promoting the
MELD-fine-tuned model to production (better on MELD, no fusion benefit, specialised away from the
webcam deployment domain).

**Remaining, in priority order:**

1. **RQ3 user study** — the empathy comparison against a text-only baseline. Needs ethics
   approval, and it is now clearly the most valuable remaining work rather than merely the last
   box to tick: §9 shows the face channel cannot be meaningfully evaluated on MELD, because MELD's
   utterance-level labels are annotated from audio + text + video and often do not describe what
   the face is doing. **The user study is the only setting in this project where the face channel
   is measured in its actual deployment domain** — real, frontal, well-lit webcam faces — and it
   is what would settle whether the multimodal system is worth the extra channel at all.
2. **A better face dataset.** FER-2013 is the remaining ceiling: 48×48, grayscale, ~10%
   mislabelled, and 8% of its test split leaks into train. AffectNet (~420k images, requires an
   access application) or RAF-DB (~30k, real-world, cleaner labels) would lift the channel far
   more than further architecture work on FER-2013 can.
3. **Frame selection over frame count.** The MELD fine-tuning samples three fixed positions per
   clip. Selecting frames where the speaker is verifiably on camera, or at peak facial motion,
   would cut label noise at its source. Untested, and the most plausible way to move (D) further.
4. **Temporal modelling** — emotion over a conversation rather than per message. The `history`
   field already reaches the backend; nothing consumes it for emotion.
5. **Demographic-bias analysis.** Face detectors have documented failure modes on darker skin
   tones. This is **flagged but unmeasured** here — no suitable test set was available — and for a
   project whose ethics section takes biometric data seriously, it is the most important gap in
   this list after RQ3.

---

## 19. Tech stack summary

- **ML / training:** Python, PyTorch, torchvision (ResNet-18), Hugging Face Transformers,
  scikit-learn, OpenCV, MediaPipe (BlazeFace), Pillow.
- **Backend:** FastAPI, uvicorn.
- **LLM:** Ollama (Qwen2.5-7B) + Google Gemini fallback.
- **Frontend:** plain HTML/JS; and Next.js 16 + React 19 + Tailwind 4 + shadcn/ui.
- **Auth + DB:** Better Auth + Drizzle ORM + Neon Postgres.
- **Testing:** pytest + a custom end-to-end harness.
- **Datasets:** GoEmotions, FER-2013, MELD.
