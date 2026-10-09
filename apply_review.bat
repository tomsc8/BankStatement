@echo off
rem runs apply_review.py with the project's virtual environment
"%~dp0.venv\Scripts\python.exe" "%~dp0apply_review.py" %*
pause
