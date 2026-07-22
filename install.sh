#!/usr/bin/env bash
# ParaClient installer — one-liner for the terminal chat client.
#
#   curl -fsSL https://raw.githubusercontent.com/paraframes-ai/paraclient-v2/main/install.sh | bash
#
# Downloads paraclient.py into an isolated venv (no clone, no touching your
# system Python) and installs a `paraclient` launcher on your PATH.
#
# Overridable via env:
#   PARACLIENT_REF=main         git ref to install from (branch/tag/sha)
#   PARACLIENT_DIR=~/.paraclient install location (holds the venv + script)
#   BIN_DIR=~/.local/bin        where the `paraclient` launcher is written
set -euo pipefail

REPO="${PARACLIENT_REPO:-paraframes-ai/paraclient-v2}"
REF="${PARACLIENT_REF:-main}"
DIR="${PARACLIENT_DIR:-$HOME/.paraclient}"
BIN_DIR="${BIN_DIR:-$HOME/.local/bin}"
RAW="https://raw.githubusercontent.com/${REPO}/${REF}"

say() { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

# --- prerequisites ---------------------------------------------------------
PY="$(command -v python3 || true)"
[ -n "$PY" ] || die "python3 not found on PATH (need 3.8+)."
"$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3,8) else 1)' \
  || die "python3 is too old — need 3.8+."

if command -v curl >/dev/null 2>&1; then
  fetch() { curl -fsSL "$1" -o "$2"; }
elif command -v wget >/dev/null 2>&1; then
  fetch() { wget -qO "$2" "$1"; }
else
  die "need curl or wget to download."
fi

# --- download --------------------------------------------------------------
say "Installing ParaClient from ${REPO}@${REF} into ${DIR}"
mkdir -p "$DIR" "$BIN_DIR"
fetch "${RAW}/paraclient.py"            "${DIR}/paraclient.py" \
  || die "could not download paraclient.py (does ref '${REF}' exist?)"
fetch "${RAW}/requirements-client.txt"  "${DIR}/requirements-client.txt" \
  || die "could not download requirements-client.txt"

# --- venv + deps -----------------------------------------------------------
say "Creating venv and installing deps (openai, rich, prompt_toolkit)"
"$PY" -m venv "${DIR}/venv"
"${DIR}/venv/bin/python" -m pip install --quiet --upgrade pip
"${DIR}/venv/bin/pip" install --quiet -r "${DIR}/requirements-client.txt"

# --- launcher --------------------------------------------------------------
LAUNCHER="${BIN_DIR}/paraclient"
cat > "$LAUNCHER" <<EOF
#!/usr/bin/env bash
exec "${DIR}/venv/bin/python" "${DIR}/paraclient.py" "\$@"
EOF
chmod +x "$LAUNCHER"

say "Installed launcher → ${LAUNCHER}"

# --- PATH hint -------------------------------------------------------------
case ":${PATH}:" in
  *":${BIN_DIR}:"*) ;;
  *) printf '\033[1;33mnote:\033[0m %s is not on your PATH. Add:\n  export PATH="%s:$PATH"\n' \
       "$BIN_DIR" "$BIN_DIR" ;;
esac

cat <<EOF

ParaClient is ready. Point it at your vLLM endpoint:

  paraclient --base-url http://localhost:8000/v1 --subject math

or set it once:

  export PARACLIENT_BASE_URL=http://localhost:8000/v1

Update later by re-running this installer. Uninstall with:  rm -rf "${DIR}" "${LAUNCHER}"
EOF
