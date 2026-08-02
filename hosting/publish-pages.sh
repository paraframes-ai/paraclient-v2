#!/usr/bin/env bash
# Publish the current installer to the GitHub Pages repos that back the custom
# download domains. Run after changing paraclient.py / install.sh / the
# requirements. Users pick up the update by re-running the one-liner.
#
#   hosting/publish-pages.sh
#
# Repos (public, Pages-enabled, custom domain via CNAME file):
#   paraframes-ai/paraclient-install-prod -> platform.prod.internal.paraframes.org
#   paraframes-ai/paraclient-install-edu  -> platform.edu.internal.paraframes.org
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

publish_one() {
  local env="$1" host="$2" repo="$3" extra_args="$4"
  local dir="$WORK/$env"
  echo "==> $repo ($host)"
  git clone -q "https://github.com/$repo.git" "$dir"
  git -C "$dir" config user.name "${GIT_AUTHOR_NAME:-AshwinVBala}"
  git -C "$dir" config user.email "${GIT_AUTHOR_EMAIL:-ashwin@csail.mit.edu}"
  cp "$ROOT/install.sh" "$ROOT/paraclient.py" "$ROOT/requirements-client.txt" "$dir/"
  echo "$host" > "$dir/CNAME"
  cat > "$dir/index.html" <<HTML
<!doctype html><meta charset=utf-8><title>ParaClient install ($env)</title>
<body style="font:16px/1.5 system-ui;max-width:40rem;margin:4rem auto;padding:0 1rem">
<h1>ParaClient${extra_args:+ ($env)}</h1>
<p>Install the terminal chat client:</p>
<pre style="background:#f4f4f4;padding:1rem;border-radius:8px;overflow:auto">curl -fsSL https://$host/install.sh | bash${extra_args}</pre>
</body>
HTML
  git -C "$dir" add -A
  if git -C "$dir" diff --cached --quiet; then
    echo "   (no changes)"
  else
    git -C "$dir" commit -q -m "Update ParaClient installer"
    git -C "$dir" push -q origin HEAD
    echo "   pushed"
  fi
}

publish_one prod platform.prod.internal.paraframes.org paraframes-ai/paraclient-install-prod ""
publish_one edu  platform.edu.internal.paraframes.org  paraframes-ai/paraclient-install-edu  " -s -- --env edu"

echo "==> Done. Live within ~1 min of the Pages rebuild."
