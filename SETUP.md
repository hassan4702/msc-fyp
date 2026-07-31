# Running Empath (setup guide)

A multimodal emotion-aware chatbot: it reads emotion from your **webcam + typed text**,
fuses the two, and replies with that in mind.

There are two ways to run it. **Option A** is the quickest — the full chatbot, no
database or Node needed.

---

## What you need

- **Python 3.12**
- **[Ollama](https://ollama.com)** — runs the reply model locally
- **The trained model weights** — these are **not in the repo** (too big for Git).
  Get `msc-fyp-weights.tar.gz` from Hashim.
- A reasonably powerful machine — the reply model is a 7B LLM and is slow on low-end hardware.
- *(Option B only)* **Node 20+ and pnpm**, plus a free **[Neon](https://neon.com)** Postgres database.

## Get the code + weights

```bash
git clone https://github.com/hassan4702/msc-fyp.git
cd msc-fyp
# extract the weights you were sent, FROM THE REPO ROOT -> creates models/weights/{text,face,mediapipe}
tar xzf /path/to/msc-fyp-weights.tar.gz
cp .env.example .env
```

`tar` ships with Windows 10+, so the same command works in PowerShell or cmd.

The paths matter. `models/weights/mediapipe/blaze_face_short_range.tflite` is hard-coded
(`backend/models/face_detect.py`) and cannot be moved with an env var — without it the face
channel refuses to start and the backend serves a placeholder that returns a constant
"neutral 0.60" forever. Check with `curl localhost:8000/health`: `"face"` must read
`CnnFaceEmotionModel`, not `StubFaceEmotionModel`.

<details><summary>Building the weights bundle (for whoever is sending it)</summary>

`models/weights` is ~2.6 GB, but 2.3 GB of that is training checkpoints the app never
loads. Ship only what inference needs — about 280 MB:

```bash
tar czf ~/msc-fyp-weights.tar.gz \
  models/weights/mediapipe \
  models/weights/face/resnet18.pt models/weights/face/calibration.json \
  models/weights/text/model.safetensors models/weights/text/config.json \
  models/weights/text/tokenizer.json models/weights/text/tokenizer_config.json \
  models/weights/text/calibration.json
```
</details>

## One-time setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-run.txt
ollama pull qwen3:4b
```

---

## Quickest: the launcher script

macOS/Linux: `./run.sh`   —   Windows: double-click `run.bat`

Both do Option A automatically: create the Python environment on first run, install
deps, point at the trained weights, and open the chatbot at http://localhost:8000.

## Windows: just double-click `run.bat`

On Windows, `run.bat` does Option A automatically: it creates the Python
environment on first run, installs deps, and starts the chatbot at
http://localhost:8000. You still need Python 3.12 installed, the weights in
`models/weights/`, and (for real replies) Ollama or a Gemini key.

## Option A — chatbot only (simplest, no login)

```bash
LLM_BACKEND=ollama \
TEXT_MODEL_DIR=models/weights/text \
FACE_MODEL_PATH=models/weights/face/resnet18.pt \
  uvicorn backend.app:app
```

Open **http://localhost:8000** — webcam + chat + live emotion detection. No database, no Node.

## Option B — full web app (login + saved chats)

Needs Node 20+/pnpm and a Neon Postgres database, in addition to Option A's backend.

1. Keep the Option A backend running (it's the emotion engine).
2. In `web/`, create a `.env` file:
   ```
   BETTER_AUTH_SECRET=   # run: openssl rand -base64 32
   BETTER_AUTH_URL=http://localhost:3000
   DATABASE_URL=         # your Neon connection string
   ```
3. Start it:
   ```bash
   cd web
   pnpm install
   pnpm drizzle-kit push     # creates the auth + chat tables in your Neon DB
   pnpm dev
   ```
   Open **http://localhost:3000**.

---

## Notes

- The webcam only works on `http://localhost` (browsers require a secure origin) — don't use the LAN IP.
- Without the LLM, the app still runs with a simple templated responder (omit `LLM_BACKEND=ollama`).
- Re-running the model **evaluation** additionally needs the datasets (GoEmotions/FER-2013/MELD) — not needed just to use the app.
