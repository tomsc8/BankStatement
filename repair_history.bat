@echo off
rem runs repair_history.py with the project's virtual environment
"%~dp0.venv\Scripts\python.exe" "%~dp0repair_history.py" %*
pause
