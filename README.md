# Multimodal Emotion-Aware Chatbot

MSc final-year project (Abertay University, CMP504 — MSc Applied AI & UX).
A chatbot that detects emotion from **face (webcam) + typed text** at the same
time, **fuses** the two signals, and uses the result to generate more empathetic
replies.

> **Status:** functional. Text model (GoEmotions, macro-F1 0.62) and face model
> (FER-2013, macro-F1 0.58) trained + temperature-calibrated; late fusion with a
> learned conflict arbiter; empathetic replies via a local LLM (Ollama). MELD
> evaluation in [evaluation/results](evaluation/results/meld_results.md). Stubs
> still ship so the repo runs with no weights/LLM. Remaining: React frontend +
> user study. See [the plan](docs/plan/2026-06-07-implementation-plan.md).

## Architecture

```
React (webcam + chat)  ->  FastAPI  ->  text model  -> P_text ┐
                                        face model  -> P_face ┼-> fusion -> emotion -> LLM -> reply
                                                                conflict?  ┘
```

Every component speaks one shared 7-emotion vocabulary (Ekman 6 + neutral), which
is what makes late fusion possible. Contracts live in `backend/models/base.py`.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install fastapi "uvicorn[standard]" pydantic httpx pytest   # skeleton only
pytest                                  # run the test suite
uvicorn backend.app:app --reload        # serve the API at http://localhost:8000
```

Try it: `POST /chat` with `{"message": "I'm fine, thanks"}` -> reply + detected emotions.
Full ML/training/eval dependencies are in `requirements.txt`.

### Run the full chatbot (trained models + local LLM)

With the trained weights present (`models/weights/`) and Ollama running:

```bash
ollama pull qwen2.5:7b                  # one-time
LLM_BACKEND=ollama \
TEXT_MODEL_DIR=models/weights/text \
FACE_MODEL_PATH=models/weights/face/face_net.pt \
  uvicorn backend.app:app
```

`POST /chat` then returns an empathetic, emotion-conditioned reply. Without these
env vars the app falls back to the keyword/stub models and a templated responder,
so it still runs anywhere.

## Layout

| Path | What |
|------|------|
| `backend/` | FastAPI app, model interfaces, fusion, responder, pipeline |
| `training/` | Scripts to fine-tune the text and face models (run on the M4 Max) |
| `evaluation/` | MELD benchmark harness + user-study analysis |
| `frontend/` | React app (webcam, chat, live emotion indicator) — to be created |
| `tests/` | Test suite |
| `docs/plan/` | The implementation plan |
