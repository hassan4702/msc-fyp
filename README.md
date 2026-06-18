# Multimodal Emotion-Aware Chatbot

A chatbot that senses **how you feel** — from both your **typed words** and your
**face on the webcam** — and replies with empathy instead of ignoring your mood.

This is an MSc final-year project (Abertay University, module CMP504, MSc Applied
AI & UX).

---

## Why we built this

Most chatbots only read your *words*. But people don't always say what they feel —
someone types *"I'm fine"* while clearly upset. A words-only bot misses that and
replies in the wrong tone.

The idea here is simple: if a chatbot can **also see your face**, it has a second
clue about your mood, and can respond more like a caring human would. This project
builds that chatbot and tests one honest question: **does adding the face actually
make the replies feel more understanding — and how should the bot behave when your
face and your words disagree?**

---

## What we built (in plain terms)

Think of it as a four-stage assembly line that runs every time you send a message:

1. **Read your words** → a language model guesses your emotion from the text.
2. **Read your face** → a vision model guesses your emotion from a webcam frame.
3. **Combine the two guesses** into a single emotion ("fusion"). This stage also
   notices when the two clues *disagree* (happy face, angry words) and decides
   which to trust.
4. **Reply with empathy** → the chosen emotion is handed to a local AI assistant,
   which writes a supportive, mood-appropriate reply.

Everything works with **7 emotions**: anger, disgust, fear, happy, sad, surprise,
and neutral.

A real example from the running system:

> **You:** "it's fine, whatever. I don't even care anymore."
> **Bot detects:** *anger* (it saw the frustration under the dismissive words)
> **Bot:** "I can really understand why you might feel that way. It must be
> frustrating and exhausting to feel like your efforts aren't being valued…"

---

## What we found so far

We measured each part on standard datasets:

- **Words alone:** about 62% accurate at naming the emotion.
- **Face alone:** about 58% accurate (faces are genuinely harder).
- **Combining them naively** (just averaging the two guesses) **did not help** —
  a weak face guess can actually drag down a good words guess.
- **A "smart referee"** that *learns* when to trust each clue was the only
  combination that didn't make things worse — and it did **best exactly when the
  face and words disagreed**, which is the interesting case.

**Honest caveat:** these numbers come from TV-show video clips, using models that
were trained on *different* data, so the scores are modest and mainly measure how
well the models *transfer* to a new setting. The real test of the face's value is
the planned study with live participants and a real webcam. Full numbers are in
[`evaluation/results/meld_results.md`](evaluation/results/meld_results.md).

---

## What's in the project

| Folder | What's inside, in plain terms |
|--------|-------------------------------|
| `backend/` | The "brain": the web service plus the code that runs the two emotion models, combines their guesses, and produces the reply. |
| `training/` | One-time scripts that *taught* the two emotion models (run on a powerful Mac). |
| `evaluation/` | The tests that *measure* how well the system works, and the saved results. |
| `frontend/` | The web page with the webcam and chat window — **planned, not built yet**. |
| `tests/` | 43 automatic checks that keep the code correct as it grows. |
| `docs/plan/` | The full written project plan. |
| `models/weights/` | The trained model files (not stored in Git — they're large). |

The system is built so it **always runs**, even with nothing installed: if the
trained models or the AI aren't present, it quietly falls back to simple
placeholder versions so you can still see the whole flow working.

---

## How to run it

**Quick look (no setup, placeholder models):**

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install fastapi "uvicorn[standard]" pydantic httpx pytest
pytest                              # run the 43 checks
uvicorn backend.app:app --reload    # start the service at http://localhost:8000
```

Then send a message: a `POST /chat` request with `{"message": "I'm fine, thanks"}`
returns a reply plus the detected emotions.

**The real chatbot (trained models + local AI):** with the trained model files in
`models/weights/` and [Ollama](https://ollama.com) running:

```bash
ollama pull qwen2.5:7b
LLM_BACKEND=ollama \
TEXT_MODEL_DIR=models/weights/text \
FACE_MODEL_PATH=models/weights/face/face_net.pt \
  uvicorn backend.app:app
```

---

## Where we are / what's next

**Done:** both emotion models (words + face), calibrating them, combining them
with the smart referee, the empathetic-reply step, and the measurement on real
data.

**Next:** the web page (webcam + chat), and a study with real people to check
whether they find the multimodal chatbot's replies more empathetic than a
words-only one.
