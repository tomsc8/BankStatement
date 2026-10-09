@echo off
rem runs audit_categories.py with the project's virtual environment
"%~dp0.venv\Scripts\python.exe" "%~dp0audit_categories.py" %*
pause
