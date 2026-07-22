# Hosting the ParaClient install endpoint

The installer downloads three files over **HTTPS from the web root** of a
platform host:

```
https://<host>/install.sh
https://<host>/paraclient.py
https://<host>/requirements-client.txt
```

Two hosts, one per environment (default is **prod**):

| Env  | Host                                      |
|------|-------------------------------------------|
| prod | `platform.prod.internal.paraframes.org`   |
| edu  | `platform.edu.internal.paraframes.org`    |

These hosts don't exist yet — this doc + `publish.sh` define what to stand up.

## Publish

```bash
hosting/publish.sh                # -> hosting/dist/{install.sh,paraclient.py,requirements-client.txt}
```

Ship `hosting/dist/` to each host's document root (rsync/scp/CI). The same
`install.sh` is served from both hosts; users pick the env with `--env edu`
(the script defaults to prod).

## Serve

Any static file server works — files must be at the root, `Content-Type`
doesn't matter (curl pipes to bash / writes to disk).

nginx:

```nginx
server {
    listen 443 ssl;
    server_name platform.prod.internal.paraframes.org;   # + edu vhost
    ssl_certificate     /etc/ssl/paraframes/platform.crt;
    ssl_certificate_key /etc/ssl/paraframes/platform.key;
    root /var/www/paraclient;                             # = hosting/dist
    location / { try_files $uri =404; }
}
```

Local smoke test before real DNS/TLS exists:

```bash
hosting/publish.sh /tmp/paraclient-www
( cd /tmp/paraclient-www && python3 -m http.server 8080 ) &
PARACLIENT_BASE=http://localhost:8080 bash install.sh   # exercises the real flow
```

## Notes

- **Internal CA:** if the hosts use an internal CA the client machine doesn't
  trust yet, users can install with `PARACLIENT_INSECURE=1` (skips TLS verify).
  Prefer distributing the CA cert instead.
- **Versioning:** root-served = "latest". To pin, serve versioned copies under
  a path and point users at it via `PARACLIENT_BASE=https://<host>/v1`.
- **Keep in sync:** re-run `publish.sh` and redeploy whenever `paraclient.py`,
  `requirements-client.txt`, or `install.sh` change in the repo.
