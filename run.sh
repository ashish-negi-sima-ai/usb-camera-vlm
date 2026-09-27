#!/usr/bin/env bash
set -euo pipefail
APP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
APP_PYTHON="${PYTHON:-${PYNEAT_ENV:-${HOME}/pyneat}/bin/python}"
if [[ ! -x "$APP_PYTHON" ]]; then
  echo "Neat Python not found: $APP_PYTHON. Set PYTHON or PYNEAT_ENV." >&2
  exit 1
fi
export PYTHONDONTWRITEBYTECODE=1
exec "$APP_PYTHON" -u "$APP_DIR/main.py" "$@"
