# Download URL via GitHub Pages (the live setup)

The one-liner is served from GitHub Pages — free, HTTPS, no server to run. Two
public repos back the two custom domains:

| Env  | Repo                                   | Custom domain                             |
|------|----------------------------------------|-------------------------------------------|
| prod | `paraframes-ai/paraclient-install-prod`| `platform.prod.internal.paraframes.org`   |
| edu  | `paraframes-ai/paraclient-install-edu` | `platform.edu.internal.paraframes.org`    |

Each repo holds only the installer files (`install.sh`, `paraclient.py`,
`requirements-client.txt`) plus a `CNAME` file naming its domain. Pages is
enabled on `master` / root.

## The one manual step: Cloudflare DNS

paraframes.org DNS is on Cloudflare, which is outside these scripts. Add two
records (proxy **off** / grey cloud, so GitHub can issue the TLS cert):

```
Type   Name                                    Target                    Proxy
CNAME  platform.prod.internal                  paraframes-ai.github.io    DNS only
CNAME  platform.edu.internal                   paraframes-ai.github.io    DNS only
```

(Enter the name however Cloudflare expects for the `internal.paraframes.org`
subtree — the full label is `platform.prod.internal.paraframes.org`.)

Within a few minutes GitHub provisions a Let's Encrypt cert for each domain and
the one-liner works from anywhere:

```bash
curl -fsSL https://platform.prod.internal.paraframes.org/install.sh | bash
curl -fsSL https://platform.edu.internal.paraframes.org/install.sh | bash -s -- --env edu
```

After DNS resolves, flip on "Enforce HTTPS" in each repo's
**Settings → Pages** (or leave it — the installer uses https already).

## Updating the client

Edit `paraclient.py` / `install.sh` / `requirements-client.txt` in this repo,
then:

```bash
hosting/publish-pages.sh
```

It copies the current files into both Pages repos and pushes. Users update by
re-running the one-liner.

## Why not the VM (`deploy/`)?

`deploy/` still works if you want a self-hosted box, but Pages needs no server,
no cert renewal, and no GCP permissions — just the two Cloudflare records above.
