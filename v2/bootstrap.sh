#!/usr/bin/env bash
# Pod entry point: dependencies, code, data, job plan, auto-stop.
# The GitHub token is never written to disk or printed: git reads it from the
# environment through an askpass helper.
set -u
export PYTHONUNBUFFERED=1
export WORK=${WORK:-/workspace}
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$WORK/logs" "$WORK/out" "$WORK/data" "$WORK/third_party"
LOG="$WORK/logs/boot_${RUNPOD_POD_ID:-local}_$(date -u +%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "[boot] $(date -u +%FT%TZ) pod=${RUNPOD_POD_ID:-?} plan=${JOBS:-}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
echo "[boot] cpus $(nproc)"; free -g | head -2

if [ -n "${GH_TOKEN:-}" ]; then
  printf '#!/bin/sh\necho "$GH_TOKEN"\n' > /root/.gh_askpass && chmod 700 /root/.gh_askpass
  export GIT_ASKPASS=/root/.gh_askpass GIT_TERMINAL_PROMPT=0
fi

stop_pod() {
  if [ "${AUTO_STOP:-1}" = "1" ] && [ -n "${RUNPOD_POD_ID:-}" ]; then
    echo "[boot] stopping pod $(date -u +%FT%TZ)"
    runpodctl stop pod "$RUNPOD_POD_ID" || echo "[boot] runpodctl stop failed"
  fi
}

if [ -n "${MAX_HOURS:-}" ]; then
  ( sleep "$(awk -v h="${MAX_HOURS}" 'BEGIN{printf "%d", h*3600}')"; echo "[watchdog] MAX_HOURS reached"; stop_pod ) &
fi

# The image ships torch in one specific interpreter; the login shell's python3
# may be a bare system Python. Pick the first interpreter that can import torch.
PY=""
for cand in python python3 /usr/bin/python3 /venv/bin/python /opt/conda/bin/python \
            /usr/local/bin/python3 /usr/bin/python3.11 /usr/bin/python3.12 /usr/bin/python3.10; do
  path="$(command -v "$cand" 2>/dev/null || true)"
  [ -z "$path" ] && [ -x "$cand" ] && path="$cand"
  [ -z "$path" ] && continue
  if "$path" -c "import torch" >/dev/null 2>&1; then PY="$path"; break; fi
done
if [ -z "$PY" ]; then
  PY="$(command -v python3)"
  echo "[boot] WARNING: no interpreter with torch found; falling back to $PY"
fi
export PY
echo "[boot] python $PY $("$PY" -V 2>&1) torch $("$PY" -c 'import torch;print(torch.__version__, torch.version.cuda)' 2>&1 | tail -1)"

"$PY" -m pip install -q --no-cache-dir pillow-heif pillow-avif-plugin einops timm scipy matplotlib 2>&1 | tail -n 3
"$PY" -c "import PIL, pillow_heif; print('[boot] pillow', PIL.__version__, 'pillow-heif', pillow_heif.__version__)"

if [ ! -d "$WORK/StegTransX-CR/.git" ]; then
  git clone -q https://github.com/Mustafa-1982/StegTransX-CR "$WORK/StegTransX-CR" \
    && git -C "$WORK/StegTransX-CR" checkout -q 44b4582f505c89f7a9cbde6ed12e4f91030120cd
fi
if [ ! -d "$WORK/third_party/StegTransX/.git" ]; then
  git clone -q https://github.com/QQ-Stars/StegTransX "$WORK/third_party/StegTransX" \
    && git -C "$WORK/third_party/StegTransX" checkout -q 8b403756439cb3e5dc9573f98f9abbdc27cb6b69
fi

cd "$REPO_DIR"
echo "[boot] repo commit $(git rev-parse --short HEAD 2>/dev/null)"

# libheif drives x265, which sizes its thread pools and frame buffers from the
# CPUs visible to the process, not from the container's cgroup quota. On a
# 256-core host a single HEIF encode of an adversarial stego image can therefore
# allocate far past the container ceiling and wedge: eval:single-cond-s0 froze at
# HEIF@50 on two different pods and two different hosts, holding 184 GiB with no
# worker pool involved, while the same commit ran other evaluations cleanly.
# x265 reads sched_getaffinity, so restricting affinity is what actually bounds
# it; nothing in pillow-heif exposes a thread count.
RUN="$PY"
if [ -n "${CPU_CAP:-}" ] && command -v taskset >/dev/null 2>&1; then
  RUN="taskset -c 0-$((CPU_CAP - 1)) $PY"
  export OMP_NUM_THREADS="$CPU_CAP"
  echo "[boot] CPU_CAP=$CPU_CAP -> $RUN"
fi
if $RUN -m exp.jobs one data; then
  $RUN -m exp.jobs plan "${JOBS}"
  echo "[boot] plan finished with code $? at $(date -u +%FT%TZ)"
else
  echo "[boot] data job failed"
fi
LOGDIR="$WORK/out/_logs_${RUNPOD_POD_ID:-local}"
mkdir -p "$LOGDIR" && cp "$WORK"/logs/*.log "$LOGDIR"/ 2>/dev/null
"$PY" -c "import os; from exp.common import push_results; p=os.environ.get('RUNPOD_POD_ID','local'); push_results('log-'+p, os.path.join(os.environ['WORK'],'out','_logs_'+p))"
stop_pod
