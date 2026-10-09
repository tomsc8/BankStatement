@echo off
rem runs train.py with the project's virtual environment
"%~dp0.venv\Scripts\python.exe" "%~dp0train.py" %*
pause
