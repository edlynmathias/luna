@echo off
REM Double-click (or type run_luna.bat) to set up the first time and start Luna.
cd /d "%~dp0"

REM onnxruntime (used by Chroma) supports Python 3.10-3.13.
set PY=
for %%v in (3.12 3.11 3.13 3.10) do (
  if not defined PY (
    py -%%v -c "import sys" >nul 2>&1 && set PY=py -%%v
  )
)
if not defined PY (
  python -c "import sys; sys.exit(not ((3,10) <= sys.version_info[:2] <= (3,13)))" >nul 2>&1 && set PY=python
)
if not defined PY (
  echo Need Python 3.10-3.13. Install Python 3.12 from https://www.python.org/downloads/ and run this again.
  pause & exit /b 1
)
echo Using %PY%

REM Remove a half-finished environment from an earlier failed install.
if exist .venv if not exist .venv\installed.txt rmdir /s /q .venv

if not exist .venv (
  echo First run: creating environment and installing packages ^(5-10 min^)...
  %PY% -m venv .venv || (echo Could not create venv & pause & exit /b 1)
  call .venv\Scripts\activate.bat
  python -m pip install --upgrade pip -q
  pip install -r requirements-eval.txt || (echo Install failed - copy the error above to Claude. & pause & exit /b 1)
  echo ok> .venv\installed.txt
) else (
  call .venv\Scripts\activate.bat
)

pip show beautifulsoup4 >nul 2>&1 || pip install -q beautifulsoup4
if not exist knowledge\web_sources.json (
  echo Building the knowledge base from trusted health websites ^(about 1 minute^)...
  python ingest.py
)
if not exist .env copy .env.example .env >nul
findstr /c:"LLM_API_KEY=replace_me" .env >nul && (
  echo.
  echo ^>^> No API key yet: answers will be source excerpts. Put your key in .env ^(LLM_API_KEY=...^) and run again.
  echo.
)
streamlit run streamlit_app.py
pause
