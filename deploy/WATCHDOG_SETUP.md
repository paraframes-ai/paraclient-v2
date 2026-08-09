# L4 Spot watchdog — setup

`l4_watchdog.sh` runs on the **e2** box (on-demand, stays up while the L4 is
down) via cron every 2 minutes. When the Spot L4 is preempted it goes
`TERMINATED` and stays there; the watchdog notices and issues `start`.

Already installed on e2:
- `~/l4_watchdog.sh`, cron: `*/2 * * * * ~/l4_watchdog.sh`
- Logs to `~/.l4_watchdog.log`.
- **Inert until a credential exists** — see below.

## The switch (so it never fights a deliberate shutdown)

Preemption and a manual `stop` both leave the VM `TERMINATED`, so the watchdog
can't tell them apart. It only acts when **not paused**:

    touch ~/.l4_watchdog_pause    # stand down — you want the L4 OFF
    rm    ~/.l4_watchdog_pause    # resume keeping it up

**To take the L4 down (e.g. overnight for cost): `touch` the pause file first,
then stop the VM.** Otherwise the watchdog restarts it within ~2 min.

## The credential it needs (one-time, in the console or your own gcloud)

The watchdog needs to call `compute.instances.{get,start}` on `paraclient-l4`.
e2's default service account lacks compute scope, so give it a dedicated,
least-privilege key:

```bash
PROJECT=paraframes-app

# 1. a service account that can do nothing else
gcloud iam service-accounts create l4-watchdog \
  --project=$PROJECT --display-name="L4 Spot watchdog"

# 2. a custom role with ONLY get/start/list (no delete, no reconfigure)
gcloud iam roles create l4Watchdog --project=$PROJECT \
  --title="L4 Watchdog" \
  --permissions=compute.instances.get,compute.instances.start,compute.instances.list

# 3. bind the role to the SA
gcloud projects add-iam-policy-binding $PROJECT \
  --member="serviceAccount:l4-watchdog@$PROJECT.iam.gserviceaccount.com" \
  --role="projects/$PROJECT/roles/l4Watchdog"

# 4. make a key
gcloud iam service-accounts keys create /tmp/l4-watchdog.json \
  --iam-account="l4-watchdog@$PROJECT.iam.gserviceaccount.com"
```

Then put the key on e2 (0600) — do it without echoing it into any transcript:

```bash
# from your machine:
scp /tmp/l4-watchdog.json  e2-gateway.tail7bcdf0.ts.net:~/.gcp/l4-watchdog.json
# on e2:
chmod 600 ~/.gcp/l4-watchdog.json && rm -f /tmp/l4-watchdog.json
```

On the next 2-minute tick the watchdog activates automatically. Confirm:

```bash
tail -f ~/.l4_watchdog.log      # should stop saying "inert"
```

Blast radius of the key: it can `get/start/list` instances in the project and
nothing else — it cannot stop, delete, or reconfigure anything.

## What it does NOT fix
Recovery still costs the VM boot + vLLM's ~3–5 min FP8 reload, so a preemption
is a ~5–8 min outage for testers. The watchdog makes recovery automatic and
fast to *trigger*; it can't remove the reload. For a guaranteed uninterrupted
window (a live demo), start the L4 as on-demand for that session instead.
