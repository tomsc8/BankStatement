#!/usr/bin/env sh
# runs audit_categories.py with the project's virtual environment
dir="$(cd "$(dirname "$0")" && pwd)"
exec "$dir/.venv/bin/python" "$dir/audit_categories.py" "$@"
