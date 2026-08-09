#!/usr/bin/env bash
# l4_watchdog.sh — keep the Spot L4 running by restarting it after a preemption.
#
# Runs on the e2 box (which is on-demand and stays up while the L4 is down).
# A Spot VM does NOT auto-restart when Google reclaims it — it goes TERMINATED
# and stays there. This notices that and issues `start`.
#
# IT MUST NOT FIGHT A DELIBERATE SHUTDOWN. Preemption and a manual `stop` both
# leave the instance TERMINATED, indistinguishable by status alone. So the
# watchdog only acts when it is ENABLED via a sentinel file:
#
#     touch  ~/.l4_watchdog_pause     -> stand down (you want the L4 OFF)
#     rm     ~/.l4_watchdog_pause     -> resume keeping it up
#
# So the workflow to take the L4 down for the night is: pause first, then stop.
#
# CREDENTIAL: needs a key that may start the instance. Point CRED at a service
# account JSON with a role granting compute.instances.{get,start} on this
# instance (see the IAM note in the deploy step). Until that file exists the
# watchdog logs "no credential" and exits 0 — inert, never error-spamming cron.
set -euo pipefail

PROJECT="${L4_PROJECT:-paraframes-app}"
ZONE="${L4_ZONE:-us-east1-b}"
INSTANCE="${L4_INSTANCE:-paraclient-l4}"
CRED="${L4_WATCHDOG_CRED:-$HOME/.gcp/l4-watchdog.json}"
PAUSE="$HOME/.l4_watchdog_pause"
LOG="$HOME/.l4_watchdog.log"
GCLOUD="$(command -v gcloud || echo /snap/bin/gcloud)"

log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" >> "$LOG"; }

# Operator asked for the box to stay down — do nothing.
[ -f "$PAUSE" ] && exit 0

# No credential yet -> stay inert (log once an hour so it isn't silent forever).
if [ ! -f "$CRED" ]; then
  if [ ! -f "$LOG" ] || [ -z "$(find "$LOG" -mmin -60 2>/dev/null)" ]; then
    log "inert: no credential at $CRED (see deploy note for the IAM to grant)"
  fi
  exit 0
fi

# Use an isolated gcloud config dir so this never disturbs interactive gcloud.
export CLOUDSDK_CONFIG="$HOME/.gcp/watchdog-sdk"
mkdir -p "$CLOUDSDK_CONFIG"
"$GCLOUD" auth activate-service-account --key-file="$CRED" --quiet >/dev/null 2>&1 || {
  log "ERROR: could not activate service account from $CRED"; exit 0; }

STATUS="$("$GCLOUD" compute instances describe "$INSTANCE" \
  --zone "$ZONE" --project "$PROJECT" --format='value(status)' 2>>"$LOG" || echo UNKNOWN)"

case "$STATUS" in
  RUNNING)                 : ;;                    # healthy, nothing to do
  PROVISIONING|STAGING|STOPPING|REPAIRING)
                           : ;;                    # a transition is in flight
  TERMINATED|SUSPENDED)
    log "instance is $STATUS and watchdog is enabled -> starting"
    if "$GCLOUD" compute instances start "$INSTANCE" --zone "$ZONE" \
         --project "$PROJECT" --quiet >>"$LOG" 2>&1; then
      log "start issued OK"
    else
      log "start FAILED (likely Spot capacity unavailable right now; will retry)"
    fi
    ;;
  *) log "unexpected status: $STATUS" ;;
esac
