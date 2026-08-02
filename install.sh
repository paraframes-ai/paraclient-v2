#!/usr/bin/env bash
# ParaClient installer — one-liner for the terminal chat client.
#
#   # prod (default)
#   curl -fsSL https://platform.prod.internal.paraframes.org/install.sh | bash
#   # edu
#   curl -fsSL https://platform.edu.internal.paraframes.org/install.sh | bash -s -- --env edu
#
# Downloads paraclient.py into an isolated venv (no clone, no touching your
# system Python) and installs a `paraclient` launcher on your PATH.
#
# Selecting the source:
#   --env prod|edu              which platform host to download from (default prod)
#   PARACLIENT_ENV=prod|edu     same, via env var
#   PARACLIENT_HOST=host        override the host entirely
#   PARACLIENT_BASE=url         override the full base URL (scheme+host[+path])
#   PARACLIENT_INSECURE=1       skip TLS verification (internal CA not trusted yet)
# Install location:
#   PARACLIENT_DIR=~/.paraclient   holds the venv + script
#   BIN_DIR=~/.local/bin           where the `paraclient` launcher is written
set -euo pipefail

ENV="${PARACLIENT_ENV:-prod}"

# --- args (support `curl ... | bash -s -- --env edu`) ----------------------
while [ $# -gt 0 ]; do
  case "$1" in
    --env)   ENV="${2:-}"; shift 2 ;;
    --env=*) ENV="${1#*=}"; shift ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) printf 'unknown arg: %s\n' "$1" >&2; exit 2 ;;
  esac
done

say() { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

case "$ENV" in prod|edu) ;; *) die "--env must be 'prod' or 'edu' (got '$ENV')" ;; esac

HOST="${PARACLIENT_HOST:-platform.${ENV}.internal.paraframes.org}"
BASE="${PARACLIENT_BASE:-https://${HOST}}"
DIR="${PARACLIENT_DIR:-$HOME/.paraclient}"
BIN_DIR="${BIN_DIR:-$HOME/.local/bin}"

# --- prerequisites ---------------------------------------------------------
PY="$(command -v python3 || true)"
[ -n "$PY" ] || die "python3 not found on PATH (need 3.8+)."
"$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3,8) else 1)' \
  || die "python3 is too old — need 3.8+."

INSECURE=""
[ "${PARACLIENT_INSECURE:-0}" = "1" ] && INSECURE=1
if command -v curl >/dev/null 2>&1; then
  fetch() { curl -fsSL ${INSECURE:+-k} "$1" -o "$2"; }
elif command -v wget >/dev/null 2>&1; then
  fetch() { wget -q ${INSECURE:+--no-check-certificate} -O "$2" "$1"; }
else
  die "need curl or wget to download."
fi

# --- download --------------------------------------------------------------
say "Installing ParaClient (${ENV}) from ${BASE} into ${DIR}"
mkdir -p "$DIR" "$BIN_DIR"
fetch "${BASE}/paraclient.py"            "${DIR}/paraclient.py" \
  || die "could not download ${BASE}/paraclient.py (is the ${ENV} host reachable?)"
fetch "${BASE}/requirements-client.txt"  "${DIR}/requirements-client.txt" \
  || die "could not download ${BASE}/requirements-client.txt"

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
