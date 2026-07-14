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
| Face detection | OpenCV Haar cascade | ❌ built-in | — |
| Face → emotion | Custom CNN (`FaceNet`) | ✅ from scratch | FER-2013 |
| Fusion arbiter | Logistic Regression | ✅ trained | MELD |
| Reply generation | Qwen2.5-7B (Ollama) / Gemini 3.5 Flash | ❌ off-the-shelf | — |

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
- **What:** a compact custom CNN (`backend/models/face_net.py`) — four conv blocks over a 48×48
  grayscale face → 7 emotions. Deliberately small to run on CPU on a weak device.
- **How:** trained from scratch on FER-2013 (via the `Aaryan333/...` HF mirror: 28,709 train /
  3,589 validation / 3,589 test, the standard FER split). PIL augmentation (flip + small
  rotation), class-weighted loss, best-on-validation checkpointing. Labels mapped **by name**
  (HF mirrors disagree on index order). Code: `training/train_face.py`, mapping in
  `training/fer2013.py`.
- **Result:** test **macro-F1 0.581**, accuracy 0.597. Per-class F1: happy 0.84, surprise 0.75,
  disgust 0.60, neutral 0.57, anger 0.51, sad 0.47, **fear 0.33** (fear is the classic FER weak
  spot). This is a legitimate from-scratch FER baseline; facial emotion in the wild is genuinely
  hard, which caps this channel.
- **At runtime:** OpenCV's Haar cascade finds the face in the webcam frame; the crop is resized
  to 48×48 and classified. If no face is detected, the channel reports "unavailable" and fusion
  falls back to text (`backend/models/face_model.py`).

### 5.3 Calibration
Neural nets are overconfident, and the two models' confidences must be **comparable** before
one can be trusted over the other. Each model's logits are **temperature-scaled** — a single
fitted number per model that softens overconfidence without changing the predicted label.
- Text: **T = 1.19** (expected calibration error 0.087 → **0.046**, nearly halved).
- Face: **T = 1.05** (already well-calibrated, ECE ~0.02).
- Code: `training/calibrate.py`. Values stored in `<model>/calibration.json`; the wrappers load
  them automatically.

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

### RQ1 — MELD test set (2,610 utterances), face-detection coverage 88.7%

| System | macro-F1 |
|--------|----------|
| text-only | **0.268** |
| face-only | 0.132 |
| majority-class | 0.093 |
| fused: weighted-average | 0.266 |
| fused: confidence-gated | 0.262 |
| fused: learned arbiter | **0.269** |

### RQ2 — conflict subset (2,025 / 2,610 = 77.6% where the channels disagree)

| System | macro-F1 |
|--------|----------|
| fused: learned arbiter | **0.263** |
| text-only | 0.256 |
| fused: weighted-average | 0.255 |
| fused: confidence-gated | 0.249 |
| face-only | 0.093 |

### Findings (stated honestly)
- **Naive fusion does not beat text-only** — adding the weak face channel *slightly degrades*
  the strong text channel.
- **Only the learned arbiter avoids degradation**, marginally exceeding text-only overall and
  **winning clearly on the conflict subset**. So a *learned* conflict policy beats naive
  averaging and beats text-alone — direct evidence for RQ2.
- **Critical caveat:** these are **cross-corpus transfer** numbers — the models were trained on
  GoEmotions / FER-2013 and applied to MELD **with no MELD fine-tuning**. That is why absolute
  F1 is ~0.27 (well below in-domain MELD text SOTA). This measures *generalisation*, not
  in-domain performance. In-domain fine-tuning is the natural follow-up.
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

- **47 unit tests** (`pytest`, in `tests/`): the label mappings, calibration, fusion strategies
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
  services/         pipeline orchestration
  app.py            FastAPI: /health, /chat, / (HTML UI)
training/       fine-tuning + data-prep scripts (run on the powerful machine)
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
ollama pull qwen2.5:7b            # optional; else it falls back to Gemini/template
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
auth and saved chats; 47 tests + an e2e harness.

**Remaining / future work:**
- **RQ3 user study** — the empathy comparison against a text-only baseline (needs ethics
  approval; the core remaining academic piece).
- **In-domain MELD fine-tuning** — to lift the channels and give in-domain (not cross-corpus)
  RQ1 numbers.
- Temporal modelling (emotion over a conversation, not per message) and demographic-bias analysis.

---

## 19. Tech stack summary

- **ML / training:** Python, PyTorch, Hugging Face Transformers, scikit-learn, OpenCV, Pillow.
- **Backend:** FastAPI, uvicorn.
- **LLM:** Ollama (Qwen2.5-7B) + Google Gemini fallback.
- **Frontend:** plain HTML/JS; and Next.js 16 + React 19 + Tailwind 4 + shadcn/ui.
- **Auth + DB:** Better Auth + Drizzle ORM + Neon Postgres.
- **Testing:** pytest + a custom end-to-end harness.
- **Datasets:** GoEmotions, FER-2013, MELD.
