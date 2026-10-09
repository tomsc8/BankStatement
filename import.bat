@echo off
rem runs import.py with the project's virtual environment
"%~dp0.venv\Scripts\python.exe" "%~dp0import.py" %*
pause
