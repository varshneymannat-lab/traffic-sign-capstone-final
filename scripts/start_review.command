#!/bin/bash
cd "$(dirname "$0")/.." || exit 1
if [ -x .venv/bin/python ]; then
  PYTHON=.venv/bin/python
elif [ -x ../.venv/bin/python ]; then
  PYTHON=../.venv/bin/python
else
  echo "Create and activate the Python environment as described in README.md first."
  exit 1
fi
"$PYTHON" scripts/restore_inference_models.py || exit 1
"$PYTHON" -m src.review_app --port 8765
