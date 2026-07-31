#!/bin/sh
# macOS/Linux equivalent of run.bat — one-command launch of the chatbot.
set -e
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
  echo "[setup] Creating Python environment (first run only)..."
  python3.12 -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
fi

# Install every run, not just the first. An existing .venv made before a dependency was
# added never gets it, and the app degrades silently: a missing torchvision drops the face
# model to the stub, which returns a constant "neutral 0.60" that looks like a real reading.
echo "[setup] Checking dependencies..."
.venv/bin/python -m pip install -q -r requirements-run.txt

export LLM_BACKEND="${LLM_BACKEND:-auto}"
export TEXT_MODEL_DIR="${TEXT_MODEL_DIR:-models/weights/text}"
export FACE_MODEL_PATH="${FACE_MODEL_PATH:-models/weights/face/resnet18.pt}"

if [ ! -f models/weights/text/model.safetensors ]; then
  echo "[note] Trained models not found in models/weights/ - running with placeholder"
  echo "       models. Extract msc-fyp-weights.tar.gz into the 'models' folder."
fi

echo "Starting the server at http://localhost:8000  (Ctrl+C to stop)"
(sleep 8 && open http://localhost:8000) &
exec .venv/bin/python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
