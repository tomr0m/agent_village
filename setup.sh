#!/usr/bin/env bash
#
# Agent Village setup: create a virtualenv, install dependencies, and prepare
# the local configuration. Safe to re-run — every step is idempotent.
#
#   ./setup.sh              full install
#   ./setup.sh --slim       skip rembg/onnxruntime (much faster; background
#                           removal then degrades to a pass-through)
#
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

VENV_DIR="${VENV_DIR:-.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SLIM=0
for arg in "$@"; do
  case "$arg" in
    --slim) SLIM=1 ;;
    -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done

say()  { printf '\033[36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[33m!! \033[0m %s\n' "$*" >&2; }
die()  { printf '\033[31mxx \033[0m %s\n' "$*" >&2; exit 1; }

# ---- 1. interpreter --------------------------------------------------------
command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "$PYTHON_BIN not found. Install Python 3.10+."

PY_VERSION="$("$PYTHON_BIN" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
PY_OK="$("$PYTHON_BIN" -c 'import sys; print(1 if sys.version_info[:2] >= (3, 10) else 0)')"
[ "$PY_OK" = "1" ] || die "Python 3.10+ required; found $PY_VERSION."
say "Using $PYTHON_BIN ($PY_VERSION)"

# ---- 2. virtualenv ---------------------------------------------------------
if [ ! -d "$VENV_DIR" ]; then
  say "Creating virtualenv in $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
else
  say "Reusing existing virtualenv $VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
say "Activated $(python -c 'import sys; print(sys.prefix)')"

# ---- 3. dependencies -------------------------------------------------------
say "Upgrading pip toolchain"
python -m pip install --quiet --upgrade pip setuptools wheel

if [ "$SLIM" = "1" ]; then
  say "Installing dependencies (slim: no rembg/onnxruntime)"
  grep -vE '^(rembg|onnxruntime)' requirements.txt > .requirements.slim.txt
  python -m pip install -r .requirements.slim.txt
  rm -f .requirements.slim.txt
  warn "Background removal will pass through until you install rembg."
else
  say "Installing dependencies (this pulls onnxruntime and takes a few minutes)"
  python -m pip install -r requirements.txt
fi

# ---- 4. configuration ------------------------------------------------------
if [ ! -f .env ]; then
  cp .env.example .env
  say "Created .env from .env.example (DRY_RUN=true)"
else
  say "Keeping your existing .env"
fi

mkdir -p storage
[ -f storage/.gitkeep ] || touch storage/.gitkeep

# ---- 5. database and smoke check ------------------------------------------
say "Initialising the database"
python main.py --init-db

say "Verifying the configuration"
python main.py --check || warn "Configuration check reported findings (see above)."

cat <<'BANNER'

  Agent Village is ready.

    source .venv/bin/activate
    python main.py --generate 1     create a listing (simulated end to end)
    python main.py --status         see what is pending
    python main.py --bot            run the Telegram approval worker
    python main.py --daemon         schedule generation and serve the bot

  DRY_RUN=true is the default: nothing reaches Printify or Etsy until you
  set DRY_RUN=false and supply PRINTIFY_API_KEY and PRINTIFY_SHOP_ID.

BANNER
