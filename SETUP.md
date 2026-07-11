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
# extract the weights you were sent -> creates models/weights/{text,face}
tar xzf /path/to/msc-fyp-weights.tar.gz -C models
```

## One-time setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-run.txt
ollama pull qwen2.5:7b
```

---

## Option A — chatbot only (simplest, no login)

```bash
LLM_BACKEND=ollama \
TEXT_MODEL_DIR=models/weights/text \
FACE_MODEL_PATH=models/weights/face/face_net.pt \
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
