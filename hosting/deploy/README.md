# Deploy the install endpoint (new e2-micro on GCP)

Stands up a small internal-only VM that serves the ParaClient installer at
`https://platform.{prod,edu}.internal.paraframes.org/`. Both hostnames resolve
to the same VM; users pick the env with `--env edu` at install time.

## One-time setup

```bash
cd hosting/deploy
cp config.env.example config.env      # fill in PROJECT/ZONE/VPC/... (config.env is gitignored)
gcloud auth login                     # if not already authenticated
./provision-vm.sh                     # creates VM + firewall (+ DNS if DNS_ZONE set)
# wait ~1 min for the boot startup-script (installs nginx + interim TLS + vhost)
./deploy.sh                           # publishes files and pushes them to the VM
```

Prereq if you set `DNS_ZONE`: a **private** Cloud DNS zone for
`internal.paraframes.org` attached to your VPC. Create once with:

```bash
gcloud dns managed-zones create paraframes-internal \
  --dns-name="internal.paraframes.org." --visibility=private \
  --networks=<your-vpc>
```

## Updating the client

Edit `paraclient.py` / `install.sh` / `requirements-client.txt`, then:

```bash
cd hosting/deploy && ./deploy.sh
```

Clients update by re-running the install one-liner — nothing else to do.

## What each file does

| file                 | role                                                        |
|----------------------|-------------------------------------------------------------|
| `config.env.example` | copy to `config.env`; environment settings for the scripts  |
| `startup-script.sh`  | runs on the VM at boot: nginx + self-signed cert + vhost     |
| `provision-vm.sh`    | creates the VM, firewall rule, and (optional) DNS A records  |
| `deploy.sh`          | `publish.sh` + push `dist/` to `/var/www/paraclient`, reload |

## TLS: read before real rollout

`startup-script.sh` generates a **self-signed** cert so HTTPS works day one.
Clients must then install with `PARACLIENT_INSECURE=1`, which skips TLS verify —
fine for a first test, **not** for production. Replace it with an internal-CA
cert for both hostnames (drop `platform.crt`/`platform.key` in
`/etc/ssl/paraframes/` and `sudo systemctl reload nginx`), and distribute the CA
to client machines so the plain one-liner verifies cleanly.
