#!/usr/bin/env bash
# Temporary Resilient VPS Runner for Google Colab CLI (Oracle 2-4 OCPU Ready)
# Isolated in vps_colab_runner/ for easy removal after training completes.
# Guards: (1) trap EXIT/INT/TERM auto-stop, (2) hard wall-clock watchdog,
# (3) NaN / EARLY REGRESSION ALARM fail-fast, (4) atomic SHA256 checkpoint sync
# & bounded auto-resume on network disconnect.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

NOTEBOOK_KEY="nb1"
NOTEBOOK_FILE=""
GPU_TYPE="L4"
SESSION_NAME=""
MAX_RUNTIME_MIN=""
MAX_RETRIES=3
SYNC_INTERVAL_SEC=300
STOP_ON_ALARM=1
OUTPUT_DIR=""
DRY_RUN=0
ACTION="run"
CRED_PATH="${GOOGLE_APPLICATION_CREDENTIALS:-$HOME/.config/gcloud/application_default_credentials.json}"
VENV_DIR="${COLAB_VENV_DIR:-$HOME/.colab-venv}"
COLAB_BIN="${COLAB_BIN:-}"

usage() {
  cat <<'EOF'
Usage: bash vps_colab_runner/run_colab_vps.sh [OPTIONS]
  --notebook nb1|nb2       Preset notebook (nb1=Stage1, nb2=Stage2) [default: nb1]
  --file <path>            Explicit notebook path (overrides preset)
  --gpu L4|T4              Colab GPU accelerator [default: L4]
  --session <name>         Colab session name [default: nb1 or nb2]
  --max-minutes <int>      Hard wall-clock budget before auto-stop [default: 210 nb1, 240 nb2]
  --max-retries <int>      Max auto-resume attempts after disconnect [default: 3]
  --sync-interval <sec>    Checkpoint sync & alarm check interval [default: 300]
  --output-dir <dir>       Local VPS backup directory [default: vps_colab_runner/runs/<session>]
  --credentials <path>     ADC JSON path [default: ~/.config/gcloud/application_default_credentials.json]
  --no-stop-on-alarm       Do not stop on EARLY REGRESSION ALARM (still stops on NaN)
  --status                 Show Colab balance, sessions, and local VPS checkpoints
  --stop-all               Immediately stop all active Colab sessions and verify 0 usage
  --dry-run                Validate config, paths, and credentials without creating any GPU VM
  -h, --help               Show this help message
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --notebook) NOTEBOOK_KEY="$2"; shift 2 ;;
    --file) NOTEBOOK_FILE="$2"; shift 2 ;;
    --gpu) GPU_TYPE="$2"; shift 2 ;;
    --session) SESSION_NAME="$2"; shift 2 ;;
    --max-minutes) MAX_RUNTIME_MIN="$2"; shift 2 ;;
    --max-retries) MAX_RETRIES="$2"; shift 2 ;;
    --sync-interval) SYNC_INTERVAL_SEC="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --credentials) CRED_PATH="$2"; shift 2 ;;
    --no-stop-on-alarm) STOP_ON_ALARM=0; shift ;;
    --status) ACTION="status"; shift ;;
    --stop-all) ACTION="stop_all"; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf '[ERROR] Unknown option: %s\n' "$1" >&2; usage; exit 2 ;;
  esac
done

if [ -z "$COLAB_BIN" ]; then
  if [ -x "$VENV_DIR/bin/colab" ]; then COLAB_BIN="$VENV_DIR/bin/colab"
  elif command -v colab >/dev/null 2>&1; then COLAB_BIN="$(command -v colab)"
  else COLAB_BIN="colab"; fi
fi

if [ "$NOTEBOOK_KEY" = "nb1" ]; then
  : "${SESSION_NAME:=nb1}"
  : "${MAX_RUNTIME_MIN:=210}"
  [ -z "$NOTEBOOK_FILE" ] && NOTEBOOK_FILE="$( [ "$GPU_TYPE" = "T4" ] && echo notebooks/01_Train_Stage1_Classifier_T4.ipynb || echo notebooks/01_Train_Stage1_Classifier_L4.ipynb )"
  CKPT_LATEST_NAME="stage1_latest.pth"
  CKPT_BEST_NAME="stage1_retrained_best.pth"
  REMOTE_ZIP="/content/SE374_Stage1_Training_Results.zip"
  REMOTE_METRICS_JSON="/content/stage1_results/STAGE1_TRAINING_METRICS.json"
  REMOTE_PLOT_PNG="/content/stage1_results/stage1_training_curves.png"
elif [ "$NOTEBOOK_KEY" = "nb2" ]; then
  : "${SESSION_NAME:=nb2}"
  : "${MAX_RUNTIME_MIN:=240}"
  [ -z "$NOTEBOOK_FILE" ] && NOTEBOOK_FILE="$( [ "$GPU_TYPE" = "T4" ] && echo notebooks/02_Pseudo_Masks_Train_Stage2_T4.ipynb || echo notebooks/02_Pseudo_Masks_Train_Stage2_L4.ipynb )"
  CKPT_LATEST_NAME="stage2_latest.pth"
  CKPT_BEST_NAME="stage2_retrained_best.pth"
  REMOTE_ZIP="/content/SE374_Stage2_Training_Results.zip"
  REMOTE_METRICS_JSON="/content/stage2_results/STAGE2_TRAINING_METRICS.json"
  REMOTE_PLOT_PNG="/content/stage2_results/stage2_training_loss.png"
else
  printf '[ERROR] Unsupported --notebook value: %s (expected nb1 or nb2)\n' "$NOTEBOOK_KEY" >&2
  exit 2
fi

: "${OUTPUT_DIR:=vps_colab_runner/runs/$SESSION_NAME}"
CKPT_LOCAL_DIR="$OUTPUT_DIR/checkpoints"
ARTIFACTS_DIR="$OUTPUT_DIR/artifacts"
RUN_LOG="$OUTPUT_DIR/run.log"
STATE_DIR="$OUTPUT_DIR/.state"

log() {
  local msg="[VPS-RUNNER $(date '+%Y-%m-%d %H:%M:%S')] $*"
  printf '%s\n' "$msg"
  [ -d "$OUTPUT_DIR" ] && printf '%s\n' "$msg" >> "$RUN_LOG" 2>/dev/null || true
}

colab_cmd() {
  GOOGLE_APPLICATION_CREDENTIALS="$CRED_PATH" "$COLAB_BIN" --auth adc "$@"
}

if [ "$ACTION" = "status" ]; then
  printf '=== Colab Usage & Sessions ===\n'
  colab_cmd usage || true
  colab_cmd sessions || true
  printf '=== Local VPS Checkpoints (%s) ===\n' "$CKPT_LOCAL_DIR"
  ls -lh "$CKPT_LOCAL_DIR" 2>/dev/null || printf '(none yet)\n'
  exit 0
fi

if [ "$ACTION" = "stop_all" ]; then
  log "Stopping all active Colab sessions..."
  colab_cmd sessions 2>/dev/null || true
  for sname in "$SESSION_NAME" nb1 nb2; do
    colab_cmd stop -s "$sname" >/dev/null 2>&1 || true
  done
  colab_cmd usage || true
  exit 0
fi

if [ "$DRY_RUN" -eq 1 ]; then
  printf '[DRY-RUN] Notebook preset : %s (%s)\n' "$NOTEBOOK_KEY" "$NOTEBOOK_FILE"
  printf '[DRY-RUN] GPU / Session   : %s / %s\n' "$GPU_TYPE" "$SESSION_NAME"
  printf '[DRY-RUN] Hard Wall-Clock : %s min | Max Retries: %s | Sync: %ss | StopOnAlarm: %s\n' \
    "$MAX_RUNTIME_MIN" "$MAX_RETRIES" "$SYNC_INTERVAL_SEC" "$STOP_ON_ALARM"
  printf '[DRY-RUN] Output Dir      : %s\n' "$OUTPUT_DIR"
  printf '[DRY-RUN] Credentials     : %s (exists=%s) | Colab: %s\n' \
    "$CRED_PATH" "$([ -f "$CRED_PATH" ] && echo yes || echo no)" "$COLAB_BIN"
  if [ ! -f "$NOTEBOOK_FILE" ]; then
    printf '[DRY-RUN ERROR] Notebook file not found: %s\n' "$NOTEBOOK_FILE" >&2
    exit 1
  fi
  printf '[DRY-RUN] Validation PASS. Zero Colab sessions created.\n'
  exit 0
fi

[ -f "$NOTEBOOK_FILE" ] || { printf '[ERROR] Notebook file not found: %s\n' "$NOTEBOOK_FILE" >&2; exit 1; }
[ -f "$CRED_PATH" ] || { printf '[ERROR] ADC credentials file not found: %s\n' "$CRED_PATH" >&2; exit 1; }
chmod 600 "$CRED_PATH" 2>/dev/null || true

mkdir -p "$CKPT_LOCAL_DIR" "$ARTIFACTS_DIR" "$STATE_DIR"
rm -f "$STATE_DIR/stop_reason" "$STATE_DIR/exec.pid"

START_EPOCH="$(date +%s)"
DEADLINE_EPOCH="$((START_EPOCH + MAX_RUNTIME_MIN * 60))"
printf 'START_EPOCH=%s\nDEADLINE_EPOCH=%s\nMAX_MINUTES=%s\nNOTEBOOK=%s\nGPU=%s\nSESSION=%s\nMAX_ATTEMPTS=%s\n' \
  "$START_EPOCH" "$DEADLINE_EPOCH" "$MAX_RUNTIME_MIN" "$NOTEBOOK_FILE" "$GPU_TYPE" "$SESSION_NAME" "$((MAX_RETRIES + 1))" > "$STATE_DIR/meta.env"
WATCHER_PID=""
EXEC_PID=""
TAIL_PID=""
CLEANED_UP=0

# Byte-exact atomic download via `colab ssh -- cat` with SHA256 verification
sync_remote_file_atomic() {
  local remote_path="$1" local_path="$2"
  local mtime_state="$STATE_DIR/$(basename "$local_path").mtime"
  local remote_meta
  remote_meta="$(colab_cmd ssh -s "$SESSION_NAME" -- stat -c '%Y %s' "$remote_path" 2>/dev/null || true)"
  [ -n "$remote_meta" ] || return 1

  if [ -f "$mtime_state" ] && [ -f "$local_path" ] && [ "$remote_meta" = "$(cat "$mtime_state" 2>/dev/null || true)" ]; then
    return 0
  fi

  local remote_sha
  remote_sha="$(colab_cmd ssh -s "$SESSION_NAME" -- sha256sum "$remote_path" 2>/dev/null | awk '{print $1}' || true)"
  [ -n "$remote_sha" ] || return 1

  local tmp_file="${local_path}.tmp.$$"
  log "Syncing $remote_path -> $local_path (meta: $remote_meta)..."
  if colab_cmd ssh -s "$SESSION_NAME" -- cat "$remote_path" > "$tmp_file"; then
    local local_sha
    local_sha="$(sha256sum "$tmp_file" | awk '{print $1}')"
    if [ "$local_sha" = "$remote_sha" ]; then
      mv -f "$tmp_file" "$local_path"
      printf '%s' "$remote_meta" > "$mtime_state"
      log "Verified atomic sync OK: $local_path (SHA256=$local_sha)"
      return 0
    fi
    log "WARN: SHA256 mismatch syncing $remote_path (expected=$remote_sha got=$local_sha); kept previous backup."
  fi
  rm -f "$tmp_file"
  return 1
}

upload_checkpoint_for_resume() {
  local local_path="$1" remote_path="$2"
  [ -f "$local_path" ] || return 0
  local local_sha
  local_sha="$(sha256sum "$local_path" | awk '{print $1}')"
  log "Uploading local checkpoint $local_path -> $remote_path for auto-resume..."
  colab_cmd ssh -s "$SESSION_NAME" -- mkdir -p /content/checkpoints
  cat "$local_path" | colab_cmd ssh -s "$SESSION_NAME" -- "cat > ${remote_path}.tmp && mv -f ${remote_path}.tmp ${remote_path}"
  local remote_sha
  remote_sha="$(colab_cmd ssh -s "$SESSION_NAME" -- sha256sum "$remote_path" 2>/dev/null | awk '{print $1}' || true)"
  if [ "$local_sha" != "$remote_sha" ]; then
    log "ERROR: Upload SHA256 mismatch for $remote_path (local=$local_sha remote=$remote_sha)"
    return 1
  fi
  log "Verified resume checkpoint uploaded: $remote_path (SHA256=$remote_sha)"
}

sync_all_checkpoints() {
  sync_remote_file_atomic "/content/checkpoints/$CKPT_BEST_NAME" "$CKPT_LOCAL_DIR/$CKPT_BEST_NAME" || true
  sync_remote_file_atomic "/content/checkpoints/$CKPT_LATEST_NAME" "$CKPT_LOCAL_DIR/$CKPT_LATEST_NAME" || true
  sync_remote_file_atomic "$REMOTE_METRICS_JSON" "$ARTIFACTS_DIR/$(basename "$REMOTE_METRICS_JSON")" || true
}

download_final_artifacts() {
  log "Downloading final deliverables from session '$SESSION_NAME'..."
  sync_all_checkpoints
  sync_remote_file_atomic "$REMOTE_PLOT_PNG" "$ARTIFACTS_DIR/$(basename "$REMOTE_PLOT_PNG")" || true
  sync_remote_file_atomic "$REMOTE_ZIP" "$ARTIFACTS_DIR/$(basename "$REMOTE_ZIP")" || true
  colab_cmd log -s "$SESSION_NAME" -o "$ARTIFACTS_DIR/${SESSION_NAME}_session_log.md" || true
  local nb_out="${NOTEBOOK_FILE%.ipynb}_output.ipynb"
  [ -f "$nb_out" ] && cp -f "$nb_out" "$ARTIFACTS_DIR/$(basename "$nb_out")"
}

trigger_stop_exec() {
  local reason="$1"
  printf '%s' "$reason" > "$STATE_DIR/stop_reason"
  sync_all_checkpoints
  [ -f "$STATE_DIR/exec.pid" ] && kill -TERM "$(cat "$STATE_DIR/exec.pid")" 2>/dev/null || true
}

start_watcher() {
  (
    local last_sync now
    last_sync="$(date +%s)"
    while true; do
      sleep 15
      now="$(date +%s)"
      if [ "$now" -ge "$DEADLINE_EPOCH" ]; then
        log "WATCHDOG TIMEOUT: Reached hard limit of ${MAX_RUNTIME_MIN} min. Saving checkpoints and stopping VM!"
        trigger_stop_exec "WATCHDOG_TIMEOUT"
        exit 0
      fi
      if [ -f "$RUN_LOG" ]; then
        if grep -qE 'Loss: nan|Non-finite epoch loss|Train Loss=nan' "$RUN_LOG"; then
          log "FAIL-FAST ALARM: Non-finite/NaN loss detected in run.log! Stopping Colab VM."
          trigger_stop_exec "NAN_ALARM"
          exit 0
        fi
        if [ "$STOP_ON_ALARM" -eq 1 ] && grep -q '\[EARLY REGRESSION ALARM\]' "$RUN_LOG"; then
          log "FAIL-FAST ALARM: [EARLY REGRESSION ALARM] triggered (Rule 17). Saving best ckpt and stopping VM."
          trigger_stop_exec "EARLY_REGRESSION_ALARM"
          exit 0
        fi
      fi
      if [ $((now - last_sync)) -ge "$SYNC_INTERVAL_SEC" ]; then
        last_sync="$now"
        sync_all_checkpoints
      fi
    done
  ) &
  WATCHER_PID="$!"
}

stop_bg() {
  local pid="$1"
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    kill -TERM "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  fi
}

cleanup() {
  [ "$CLEANED_UP" -eq 1 ] && return 0
  CLEANED_UP=1
  stop_bg "$WATCHER_PID"
  stop_bg "$EXEC_PID"
  stop_bg "$TAIL_PID"
  log "CLEANUP GUARD: Ensuring Colab session '$SESSION_NAME' is terminated to protect compute units..."
  colab_cmd stop -s "$SESSION_NAME" >/dev/null 2>&1 || true
  log "Post-teardown Colab usage check:"
  colab_cmd usage || true
}

trap cleanup EXIT INT TERM

log "Starting resilient Colab run: nb=$NOTEBOOK_FILE gpu=$GPU_TYPE session=$SESSION_NAME max_min=$MAX_RUNTIME_MIN retries=$MAX_RETRIES sync=${SYNC_INTERVAL_SEC}s"
colab_cmd usage || true

ATTEMPT=1
MAX_ATTEMPTS="$((MAX_RETRIES + 1))"
SUCCEEDED=0

while [ "$ATTEMPT" -le "$MAX_ATTEMPTS" ]; do
  if [ "$(date +%s)" -ge "$DEADLINE_EPOCH" ]; then
    log "Hard wall-clock limit reached before attempt $ATTEMPT. Exiting cleanly."
    break
  fi

  printf '%s' "$ATTEMPT" > "$STATE_DIR/attempt"
  log "=== Attempt $ATTEMPT/$MAX_ATTEMPTS: Provisioning Colab session '$SESSION_NAME' ($GPU_TYPE) ==="
  colab_cmd stop -s "$SESSION_NAME" >/dev/null 2>&1 || true
  colab_cmd new -s "$SESSION_NAME" --gpu "$GPU_TYPE"

  if [ -f "$CKPT_LOCAL_DIR/$CKPT_LATEST_NAME" ] || [ -f "$CKPT_LOCAL_DIR/$CKPT_BEST_NAME" ]; then
    log "Found saved checkpoints on VPS disk. Uploading to '$SESSION_NAME' for automatic resume..."
    upload_checkpoint_for_resume "$CKPT_LOCAL_DIR/$CKPT_LATEST_NAME" "/content/checkpoints/$CKPT_LATEST_NAME"
    upload_checkpoint_for_resume "$CKPT_LOCAL_DIR/$CKPT_BEST_NAME" "/content/checkpoints/$CKPT_BEST_NAME"
  fi

  start_watcher
  touch "$RUN_LOG"
  tail -n 0 -f "$RUN_LOG" &
  TAIL_PID="$!"
  set +e
  colab_cmd exec -s "$SESSION_NAME" -f "$NOTEBOOK_FILE" --timeout 0 >> "$RUN_LOG" 2>&1 &
  EXEC_PID="$!"
  printf '%s' "$EXEC_PID" > "$STATE_DIR/exec.pid"
  wait "$EXEC_PID"
  EXEC_RC=$?
  set -e
  EXEC_PID=""
  rm -f "$STATE_DIR/exec.pid"
  stop_bg "$TAIL_PID"; TAIL_PID=""
  stop_bg "$WATCHER_PID"; WATCHER_PID=""

  if [ -f "$STATE_DIR/stop_reason" ]; then
    log "Halting loop due to safety guard: $(cat "$STATE_DIR/stop_reason")"
    break
  fi

  if [ "$EXEC_RC" -eq 0 ]; then
    log "Notebook execution completed with exit code 0!"
    download_final_artifacts
    SUCCEEDED=1
    break
  fi

  log "WARN: Attempt $ATTEMPT exited with code $EXEC_RC (possible WebSocket/network drop). Salvaging checkpoints..."
  sync_all_checkpoints
  colab_cmd stop -s "$SESSION_NAME" >/dev/null 2>&1 || true

  if [ "$ATTEMPT" -lt "$MAX_ATTEMPTS" ]; then
    log "Waiting 15 seconds before auto-resume attempt $((ATTEMPT + 1))/$MAX_ATTEMPTS..."
    sleep 15
  fi
  ATTEMPT="$((ATTEMPT + 1))"
done

if [ "$SUCCEEDED" -eq 1 ]; then
  log "SUCCESS: All deliverables saved in $OUTPUT_DIR. Cleanup trap will now terminate '$SESSION_NAME'."
  exit 0
fi
log "STOPPED WITHOUT FULL COMPLETION. Latest salvaged checkpoints are in $CKPT_LOCAL_DIR."
exit 1
