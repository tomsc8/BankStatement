#!/usr/bin/env sh
# runs apply_review.py with the project's virtual environment
dir="$(cd "$(dirname "$0")" && pwd)"
exec "$dir/.venv/bin/python" "$dir/apply_review.py" "$@"
