#!/usr/bin/env sh
# runs train.py with the project's virtual environment
dir="$(cd "$(dirname "$0")" && pwd)"
exec "$dir/.venv/bin/python" "$dir/train.py" "$@"
