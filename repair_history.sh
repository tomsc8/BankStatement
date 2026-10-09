#!/usr/bin/env sh
# runs repair_history.py with the project's virtual environment
dir="$(cd "$(dirname "$0")" && pwd)"
exec "$dir/.venv/bin/python" "$dir/repair_history.py" "$@"
