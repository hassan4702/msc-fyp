@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo(
echo  ==================================================
echo    Empath - Multimodal Emotion-Aware Chatbot
echo  ==================================================
echo(

REM ---- 1. find Python ----
where py >nul 2>&1
if %errorlevel%==0 (
  set "PY=py -3"
) else (
  where python >nul 2>&1
  if !errorlevel!==0 (
    set "PY=python"
  ) else (
    echo  [ERROR] Python is not installed.
    echo  Install Python 3.12 from https://www.python.org/downloads/
    echo  ^(tick "Add Python to PATH" during install^) then run this again.
    echo(
    pause
    exit /b 1
  )
)

REM ---- 2. one-time environment setup ----
if not exist ".venv\Scripts\python.exe" (
  echo  [setup] Creating Python environment ^(first run only^)...
  %PY% -m venv .venv
  ".venv\Scripts\python.exe" -m pip install --upgrade pip
)

REM Install every run, not just the first. An existing .venv made before a dependency was
REM added never gets it, and the app degrades silently: a missing torchvision drops the
REM face model to the stub, which returns a constant "neutral 0.60" that looks like a real
REM reading. pip is near-instant when everything is already satisfied.
echo  [setup] Checking dependencies...
".venv\Scripts\python.exe" -m pip install -q -r requirements-run.txt

REM ---- 3. config: trained models + auto LLM (Ollama, else Gemini, else template) ----
set "LLM_BACKEND=auto"
set "TEXT_MODEL_DIR=models\weights\text"
set "FACE_MODEL_PATH=models\weights\face\resnet18.pt"

if not exist "models\weights\face\resnet18.pt" (
  echo(
  echo  [note] Face model not found - the face channel will report a placeholder.
  echo         Check GET /health: "face" should read CnnFaceEmotionModel, not Stub.
  echo(
)

if not exist "models\weights\text\model.safetensors" (
  echo(
  echo  [note] Trained models not found in models\weights\ - the app will run with
  echo         basic placeholder models. Get msc-fyp-weights.tar.gz from Hashim and
  echo         extract it into the "models" folder to use the real trained models.
  echo(
)

REM ---- 4. run ----
echo  Starting the server...
echo  Your browser will open at:  http://localhost:8000
echo  Keep this window open. Press Ctrl+C to stop.
echo(
start "" cmd /c "timeout /t 8 >nul & start http://localhost:8000"
".venv\Scripts\python.exe" -m uvicorn backend.app:app --host 127.0.0.1 --port 8000

endlocal
