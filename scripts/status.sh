#!/usr/bin/env bash
# One-glance status of every run. Usage:  bash scripts/status.sh   (or: watch -n 30 bash scripts/status.sh)
set -uo pipefail
cd "$(dirname "$0")/.."
R="$HOME/JasonLucas/outputs"; L="$HOME/JasonLucas/logs"
PY="$HOME/JasonLucas/envs/ttlm/bin/python"

echo "############ ALL JOBS ($(date +%H:%M:%S)) ############"
squeue -u "$USER" -o "%.9i %.26j %.2t %.8M %R" 2>/dev/null

echo; echo "############ OPEN SWEEP ############"
"$PY" scripts/monitor_runs.py --job-prefix full- --results-root "$R/results_full" 2>/dev/null | sed -n '3,40p'

echo; echo "############ CLOSED (gpt-4o-mini + haiku) ############"
"$PY" scripts/monitor_runs.py --job-prefix closed- --results-root "$R/results_closed_full" 2>/dev/null | sed -n '3,40p'

echo; echo "############ TAU-BENCH ############"
if [ -f "$R/results_tau/stage_cd_tau.json" ]; then echo "  DONE -> $R/results_tau/stage_cd_tau.json"
else "$PY" scripts/monitor_runs.py --job-prefix tau- --results-root "$R/results_tau" 2>/dev/null | sed -n '3,12p'; fi

echo; echo "############ DISAAD TRAINING ############"
sq=$(squeue -u "$USER" -h -o "%j %T %M" 2>/dev/null | grep disaad || echo "(no disaad job in queue)")
echo "  job: $sq"
d=$(ls -t "$L"/disaad-train-qwen3-*.err 2>/dev/null | head -1)
[ -n "$d" ] && tr '\r' '\n' < "$d" 2>/dev/null | grep -oE "Generating responses: *[0-9]+%[^]]*[0-9]+/[0-9]+[^]]*" | tail -1 | sed 's/^/  data: /'
[ -f "$R/disaad/proxy_qwen3-0.6b_from_qwen3-8b/training_manifest.json" ] && echo "  PROXY READY (manifest written)" || echo "  proxy: not yet trained"
