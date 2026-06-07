# Multimodal Emotion-Aware Chatbot — Implementation Plan

> **For agentic workers:** this is the **master plan**. Each subsystem (§5) gets
> its own detailed task-by-task TDD plan via `superpowers:writing-plans` when it
> is built. Steps here are tracked with checkboxes (`- [ ]`).

**Goal:** Build and evaluate a chatbot that detects emotion from webcam face +
typed text simultaneously, fuses the two signals, and uses the fused emotion to
generate more empathetic replies — then prove (a) fusion beats either channel
alone and (b) users find the multimodal bot more empathetic.

**Architecture:** Late fusion. A face model and a text model independently emit a
probability distribution over one shared 7-emotion label space; a fusion layer
combines them (confidence gating + a conflict policy) into a single emotion; that
emotion is injected into an LLM's system prompt to shape the reply. Backend is
FastAPI; frontend is React; the LLM sits behind a swappable interface so it can
run locally (Ollama on the M4 Max) or via a hosted API (for the weak device).

**Tech stack:** Python, PyTorch, Hugging Face Transformers, OpenCV/MediaPipe,
DeepFace, FastAPI, React, Ollama, scikit-learn, SciPy.

**Project context:** Abertay University, module CMP504, MSc Applied AI & UX,
supervisor Stuart Anderson. ~16 weeks. Risk assessment signed (ethics + UK GDPR
already acknowledged).

---

## 1. Hard constraints (these shape every decision)

1. **Two machines.** The **M4 Max / 36 GB is temporary** — used for *training and
   fine-tuning only*. The **actual deployment device is weak and cannot run
   models**. Therefore: train everything now, save artifacts; keep the live system
   light; put the only heavy component (the LLM) behind a swappable interface.
2. **No paired typed-text + webcam dataset exists.** RQ1 is therefore evaluated on
   **MELD** (multimodal: video + transcript + gold 7-class label). The "typed
   text vs face mismatch" premise is tested only in the **live user study**.
3. **Ethics + UK GDPR.** Webcam/biometric data → ethics approval **before** any
   participant. Raw video is processed in memory and never persisted or committed.
4. **Shared label space.** Ekman 6 + neutral (`backend/emotions.py`). Both models
   must output over exactly these 7, in this order, or fusion is meaningless.

## 2. Research questions → what proves each

| RQ | Question | How it is answered | Success criterion |
|----|----------|--------------------|-------------------|
| RQ1 | Does face+text beat either alone? | Macro-F1 of fused vs text-only vs face-only vs majority, on MELD-test | Fused macro-F1 > both single-channel baselines |
| RQ2 | How to handle conflicting signals? | Conflict-case analysis on MELD; compare fusion policies (avg vs gating vs learned arbiter) | A defined, evaluated policy that beats naive averaging on conflict cases |
| RQ3 | Do users find it more empathetic? | Paired user study, Likert ratings, Wilcoxon signed-rank | Statistically significant preference (p < 0.05) or a documented, honest null result |

## 3. Repo structure (created)

```
backend/        FastAPI app + model interfaces + fusion + responder + pipeline
  emotions.py     canonical 7-label space (locked)
  models/base.py  contracts: EmotionPrediction, EmotionModel, FusionStrategy, Responder
  models/         text_model, face_model, fusion, llm  (stubs now -> real later)
  services/       pipeline orchestration
  app.py          FastAPI: /health, /chat
training/       offline fine-tuning scripts (run on the M4 Max)
evaluation/     MELD benchmark + user-study analysis
frontend/       React app (Phase 4)
tests/          pytest suite
docs/plan/      this file
```

## 4. Build order (each step is independently testable before the next)

1. ✅ Walking skeleton: contracts + stub models + FastAPI + tests (this commit).
2. Text model (easiest, fully testable in isolation).
3. Face model.
4. Calibrate both → comparable confidences.
5. Fusion: weighted average → confidence gating → learned arbiter.
6. LLM behind the swappable responder interface.
7. Frontend (webcam + chat + emotion indicator).
8. MELD evaluation harness (answers RQ1/RQ2).
9. Ethics approval → user study → analysis (answers RQ3).
10. Write-up.

## 5. Subsystems — responsibility, approach, and what to test

### 5.1 Text emotion model  → replaces `StubTextEmotionModel`
- **Approach:** fine-tune `distilbert-base-uncased` on GoEmotions; map the 27
  GoEmotions labels to the 7 Ekman labels using the **official Ekman mapping**
  published by the GoEmotions authors (citable — avoids "arbitrary mapping"
  criticism). Train with AdamW, lr ≈ 2e-5, early-stop on validation macro-F1,
  class weighting for imbalance. Use ISEAR only as an out-of-distribution test.
- **What to test:** (unit) `predict()` returns a valid 7-way distribution summing
  to 1; (model) macro-F1 on GoEmotions-test reported with per-class breakdown;
  (regression) a few hand-written sentences map to the expected emotion.
- **Done when:** macro-F1 logged on a held-out split + weights exported + the
  class drops into the pipeline with no caller changes.

### 5.2 Face emotion model  → replaces `StubFaceEmotionModel`
- **Approach:** baseline with DeepFace first; then transfer-learn a
  MobileNet/ResNet on FER-2013 (request AffectNet access early — it is an
  application, not a download). Heavy augmentation (flip/rotate/brightness) for
  lighting robustness; class weighting (FER is "happy"-heavy). Runtime: detect
  face (MediaPipe), crop/align, sample ~2–5 fps, **average probabilities over a
  ~1–2 s window** at send time. Emit an `available=False` prediction when no face
  is detected.
- **What to test:** (unit) no-face input → `available=False`; (model) macro-F1 +
  confusion matrix on FER-2013-test; (robustness) accuracy under simulated low
  light / occlusion.
- **Done when:** macro-F1 logged + graceful no-face handling + drops into pipeline.

### 5.3 Calibration
- **Approach:** temperature scaling on each model's logits so `confidence` means
  the same thing across modalities (required before confidence gating is valid).
- **What to test:** expected calibration error (ECE) drops post-scaling.

### 5.4 Fusion  → `backend/models/fusion.py` (skeleton has avg + gating)
- **Approach:** three strategies of increasing power: (1) weighted average
  (tune `w` on MELD-val); (2) confidence gating + availability fallback +
  conflict flag (shipped); (3) **learned arbiter** — a logistic regression / tiny
  MLP over `[P_text, P_face, conf_text, conf_face]`, trained on MELD, that decides
  the final label especially in conflict cases.
- **What to test:** (unit, shipped) blending, fallback when a modality is
  unavailable, conflict flag on confident disagreement, no conflict on agreement;
  (eval) each strategy's macro-F1 on MELD and accuracy specifically on conflict
  cases.
- **Done when:** the arbiter beats averaging on MELD conflict cases (RQ2).

### 5.5 Responder (LLM)  → `backend/models/llm.py`
- **Approach:** prompt injection — fused emotion (+ conflict note) goes into the
  system prompt. Backends behind one interface: `TemplateResponder` (offline,
  shipped), `OllamaResponder` (local quantized 7B on the M4 Max, shipped),
  `ApiResponder` (hosted, Phase 4 — for the weak device). **Only the emotion
  label + text reach the LLM, never frames.**
- **What to test:** (unit) prompt contains the emotion and the conflict note when
  conflicted; (integration) Ollama backend returns a non-empty reply when running.
- **Done when:** swapping backend changes nothing else; latency measured on both.

### 5.6 API + pipeline  → `backend/app.py`, `backend/services/pipeline.py` (shipped)
- **What to test:** (shipped) `/health` ok; `/chat` returns reply + all three
  emotion views; text-only path when no frames.

### 5.7 Frontend (React)
- **Approach:** Vite + React; `getUserMedia` webcam; sample frames, send on
  message; chat window; live emotion indicator from the API's emotion views.
- **What to test:** webcam permission handling; text-only fallback when camera
  denied; manual UX walkthrough.

## 6. Evaluation plan (detail)

- **Datasets:** MELD (RQ1/RQ2). FER-2013 + GoEmotions for branch training; ISEAR
  as an extra text test set. AffectNet optional (access permitting).
- **Metric:** macro-averaged F1 across the 7 classes (handles imbalance), plus
  per-class F1 and confusion matrices. **Set expectations: MELD macro-F1 is
  genuinely hard — a mid-range score is normal, not failure.**
- **Baselines:** text-only, face-only, majority-class, fused.
- **Domain gap:** the FER-trained face model on MELD frames will drop; optionally
  fine-tune on MELD frames to close it — report both.
- **User study (RQ3):** within-subject, both conditions per participant, **order
  counterbalanced**; Likert (empathy / appropriateness / understanding);
  `scipy.stats.wilcoxon`. **Controls:** keep responses comparable so a preference
  reflects *correct* emotion use, not just "mentions feelings more"; blind which
  system is which; ~20 participants (proceed with fewer + document if needed).

## 7. 16-week schedule (milestones + verification)

| Phase | Weeks | Deliverable | Verify |
|-------|-------|-------------|--------|
| 1 | 1–3 | Lit review; **start ethics approval**; datasets downloaded (AffectNet request in); metrics fixed; this skeleton | tests pass; ethics submitted |
| 2 | 4–7 | Face model trained + benchmarked; failure cases handled | macro-F1 + confusion matrix logged |
| 3 | 6–9 | Text model fine-tuned + benchmarked; DistilBERT vs BERT | macro-F1 logged; speed compared |
| 4 | 9–12 | Calibration + fusion (incl. arbiter); LLM wired; FastAPI+React integrated | end-to-end demo runs; conflict policy evaluated |
| 5 | 12–16 | MELD quantitative results; user study; final report | RQ1/RQ2 numbers; Wilcoxon result; thesis |

## 8. Risk register (from the signed assessment + technical additions)

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Ethics approval slips | High | Submit week 1, in parallel with lit review |
| AffectNet access denied/slow | Medium | Proceed on FER-2013; AffectNet is a bonus, not a dependency |
| Face accuracy low in the wild | High | Augmentation, temporal averaging, confidence gating to text |
| Conflict averaging gives mush | High | Conflict policy / learned arbiter instead of plain averaging |
| Deployment device too weak for LLM | High | LLM behind swappable interface; hosted API for deployment |
| Can't recruit 20 participants | Medium | Proceed with fewer; document as a limitation |
| Novelty claim challenged in viva | Medium | Narrow to incongruent-affect + live-LLM slice; cite empathetic-dialogue prior work |
| Lab participants stay flat-faced | Medium | Light elicitation tasks; report acted-vs-spontaneous caveat |

## 9. Definition of done (deliverables)

- [ ] Working integrated system (repo) — face+text→fusion→LLM→reply, runnable on the weak device.
- [ ] Trained + benchmarked text and face models (weights + metrics).
- [ ] Evaluated fusion incl. conflict policy (MELD results: RQ1, RQ2).
- [ ] User study: ethics approval, consent forms, questionnaire, analysed results (RQ3).
- [ ] Final dissertation.
- [ ] Demo / viva.

---

## Self-review (spec coverage)

- Every RQ maps to an evaluation in §2/§6. ✅
- Every breakdown component (face, text, fusion, LLM, FastAPI, React, datasets,
  phases, baselines, user study) maps to a subsystem in §5 and the schedule in §7. ✅
- Hardware reality, ethics/GDPR, and the no-paired-data gap are all captured as
  hard constraints in §1 and risks in §8. ✅
- Gaps surfaced earlier (label-space mismatch, averaging-vs-conflict, calibration,
  bias, lab-affect, novelty) are each assigned an owner subsystem or risk. ✅
