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
  echo  [setup] Installing dependencies ^(a few minutes, one time^)...
  ".venv\Scripts\python.exe" -m pip install --upgrade pip
  ".venv\Scripts\python.exe" -m pip install -r requirements-run.txt
)

REM ---- 3. config: trained models + auto LLM (Ollama, else Gemini, else template) ----
set "LLM_BACKEND=auto"
set "TEXT_MODEL_DIR=models\weights\text"
set "FACE_MODEL_PATH=models\weights\face\face_net.pt"

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
