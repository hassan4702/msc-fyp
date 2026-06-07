# Multimodal Emotion-Aware Chatbot

MSc final-year project (Abertay University, CMP504 — MSc Applied AI & UX).
A chatbot that detects emotion from **face (webcam) + typed text** at the same
time, **fuses** the two signals, and uses the result to generate more empathetic
replies.

> **Status:** walking skeleton. Real architecture and contracts are in place;
> the emotion models and LLM are stubs that run offline so the whole pipeline
> works end-to-end today. See [the implementation plan](docs/plan/2026-06-07-implementation-plan.md).

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

## Layout

| Path | What |
|------|------|
| `backend/` | FastAPI app, model interfaces, fusion, responder, pipeline |
| `training/` | Scripts to fine-tune the text and face models (run on the M4 Max) |
| `evaluation/` | MELD benchmark harness + user-study analysis |
| `frontend/` | React app (webcam, chat, live emotion indicator) — to be created |
| `tests/` | Test suite |
| `docs/plan/` | The implementation plan |
