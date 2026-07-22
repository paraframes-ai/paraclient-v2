#!/usr/bin/env bash
# Assemble the files ParaClient's installer expects at a host's web root.
#
#   hosting/publish.sh [OUT_DIR]        # default OUT_DIR = hosting/dist
#
# Copies install.sh, paraclient.py and requirements-client.txt into OUT_DIR.
# Then serve OUT_DIR at the web root of each platform host so that:
#   https://platform.prod.internal.paraframes.org/install.sh   (and edu)
# resolves. See hosting/HOSTING.md for nginx / dev-server examples and rollout.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$REPO_ROOT/hosting/dist}"

mkdir -p "$OUT"
for f in install.sh paraclient.py requirements-client.txt; do
  cp "$REPO_ROOT/$f" "$OUT/$f"
done
chmod +x "$OUT/install.sh"

printf 'Published to %s:\n' "$OUT"
ls -l "$OUT"
printf '\nServe this dir at each host root, then test:\n'
printf '  curl -fsSL https://platform.prod.internal.paraframes.org/install.sh | bash\n'
