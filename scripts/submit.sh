#!/usr/bin/env bash
# Thin sbatch wrapper: applies the four fixed Rockfish flags this account always needs,
# and passes everything else through verbatim. So you never retype them or forget one.
#
#   Fixed (never change):  --partition, --account, --comment, --reservation
#   You supply per job:    --job-name, --gpus (omit for CPU), --time, --output, --error, script
#
# Usage:
#   scripts/submit.sh --job-name=smoke --gpus=1 --time=00:30:00 \
#       --output=$WORK/logs/%x-%j.out --error=$WORK/logs/%x-%j.err my_job.slurm [args...]
#
#   scripts/submit.sh --dry-run ...      # print the full sbatch command, submit nothing
#   NO_RESERVATION=1 scripts/submit.sh ...   # force-drop --reservation (general a100 pool)
#
# The reservation is included only when it is currently ACTIVE. This wrapper QUERIES
# `scontrol show res` at submit time, so once "JSALT 2026" expires the flag is dropped
# automatically and the job queues against the general pool -- no date to edit by hand.
set -euo pipefail

PARTITION="a100"
ACCOUNT="jsalt2026-lgarci27"
COMMENT="accept_cost"
RESERVATION="JSALT 2026"

DRY_RUN=0
PASSTHROUGH=()
for arg in "$@"; do
  if [[ "$arg" == "--dry-run" ]]; then DRY_RUN=1; else PASSTHROUGH+=("$arg"); fi
done

if [[ ${#PASSTHROUGH[@]} -eq 0 ]]; then
  echo "error: nothing to submit. Pass sbatch flags and a job script." >&2
  echo "  e.g. scripts/submit.sh --job-name=smoke --gpus=1 --time=00:30:00 \\" >&2
  echo "         --output=\$WORK/logs/%x-%j.out --error=\$WORK/logs/%x-%j.err my_job.slurm" >&2
  exit 2
fi

# Decide whether the reservation is usable right now.
use_reservation=1
if [[ "${NO_RESERVATION:-0}" == "1" ]]; then
  use_reservation=0
  echo "note: NO_RESERVATION=1 set -- submitting to the general a100 pool." >&2
else
  # Query the reservation, but never let a slow/busy controller hang the submit: cap it
  # with `timeout`. Only DROP the reservation when the query positively reports a
  # non-ACTIVE state; an empty result (timeout/unavailable) keeps it -- the reservation is
  # the intended path, and sbatch will fail fast if it has actually expired.
  res_state="$(timeout 15 scontrol show res "$RESERVATION" 2>/dev/null | grep -oP 'State=\K\S+' || true)"
  if [[ -z "$res_state" ]]; then
    echo "warning: could not verify reservation '$RESERVATION' (scontrol slow/unavailable)." >&2
    echo "         Keeping --reservation; sbatch will reject fast if it has expired." >&2
  elif [[ "$res_state" != "ACTIVE" ]]; then
    use_reservation=0
    echo "warning: reservation '$RESERVATION' is not ACTIVE (state='$res_state')." >&2
    echo "         Dropping --reservation; job will queue against the general a100 pool." >&2
  fi
fi

CMD=(sbatch
  "--partition=${PARTITION}"
  "--account=${ACCOUNT}"
  "--comment=${COMMENT}")
if [[ "$use_reservation" == "1" ]]; then
  CMD+=("--reservation=${RESERVATION}")
fi
CMD+=("${PASSTHROUGH[@]}")

if [[ "$DRY_RUN" == "1" ]]; then
  printf 'DRY RUN -- would submit:\n'
  printf '  %q' "${CMD[@]}"; printf '\n'
  exit 0
fi

exec "${CMD[@]}"
