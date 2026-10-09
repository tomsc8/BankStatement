#!/usr/bin/env sh
# runs import.py with the project's virtual environment
dir="$(cd "$(dirname "$0")" && pwd)"
exec "$dir/.venv/bin/python" "$dir/import.py" "$@"
