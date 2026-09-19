@echo off
title Context Aware Bangla Text Analyzer (CSE 4121)
echo ======================================================================
echo Launching Context Aware Bangla Text Analyzer (Streamlit App)...
echo Dept. of CSE, KUET
echo ======================================================================
echo.

if exist "..\.venv\Scripts\streamlit.exe" (
    "..\.venv\Scripts\streamlit.exe" run "Project files/app.py"
) else if exist ".venv\Scripts\streamlit.exe" (
    ".venv\Scripts\streamlit.exe" run "Project files/app.py"
) else (
    streamlit run "Project files/app.py"
)

pause
