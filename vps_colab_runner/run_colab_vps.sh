#!/usr/bin/env bash
# Resilient P0 ComfyUI-Safe Multi-Session & Multi-GPU VPS Runner for Google Colab CLI.
# Guards: (1) account-level provision flock + immutable protected_endpoints snapshot,
# (2) owned-only dual-match stop + partial-provision cleanup, (3) per-session --config & notebook copy,
# (4) watchdog + NaN/collapse alarm, (5) post-watcher SHA256 checkpoint sync & bounded auto-resume.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

NOTEBOOK_KEY="nb1"; NOTEBOOK_FILE=""; GPU_TYPE="L4"; GPU_FALLBACK=""; HIGH_MEM="auto"
SESSION_NAME=""; PRESET_NAME=""; MAX_RUNTIME_MIN=""; MAX_RETRIES=3; SYNC_INTERVAL_SEC=300
STOP_ON_ALARM=1; OUTPUT_DIR=""; CUSTOM_OUT_DIR=0; DRY_RUN=0; ACTION="run"; TARGET_STOP_SESSION=""
CRED_PATH="${GOOGLE_APPLICATION_CREDENTIALS:-$HOME/.config/gcloud/application_default_credentials.json}"
VENV_DIR="${COLAB_VENV_DIR:-$HOME/.colab-venv}"; COLAB_BIN="${COLAB_BIN:-}"
PY_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"
ENV_ARGS=(); ENV_DISPLAY=()

usage() {
  cat <<'EOF'
Usage: bash vps_colab_runner/run_colab_vps.sh [OPTIONS]
  --notebook nb1|nb2 | --file <path> | --gpu T4|L4|G4|A100|H100 | --gpu-fallback <csv>
  --high-mem | --no-high-mem (auto-on H100/G4; default 0 on A100/L4/T4)
  --env KEY=VALUE | --preset <name> | --session <name> (^se374_... or nb1|nb2)
  --max-minutes <int> [210/240] | --max-retries <int> [3] | --sync-interval <sec> [300]
  --output-dir <dir> | --credentials <path> | --no-stop-on-alarm | --dry-run | -h
  --status | --stop-session <name> | --stop-owned | --stop-all
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --notebook) NOTEBOOK_KEY="$2"; shift 2 ;;
    --file) NOTEBOOK_FILE="$2"; shift 2 ;;
    --gpu) GPU_TYPE="$(printf '%s' "$2" | tr '[:lower:]' '[:upper:]')"; shift 2 ;;
    --high-mem) HIGH_MEM=1; shift ;;
    --no-high-mem) HIGH_MEM=0; shift ;;
    --gpu-fallback) GPU_FALLBACK="$(printf '%s' "$2" | tr '[:lower:]' '[:upper:]')"; shift 2 ;;
    --env)
      [[ "$2" == *=* ]] || { printf '[ERROR] --env expects KEY=VALUE, got: %s\n' "$2" >&2; exit 2; }
      [[ "${2%%=*}" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || { printf '[ERROR] Invalid --env key: %s\n' "${2%%=*}" >&2; exit 2; }
      ENV_ARGS+=(--env "$2"); ENV_DISPLAY+=("$2"); shift 2 ;;
    --preset) PRESET_NAME="$2"; ENV_ARGS+=(--env "STAGE1_EXP_PRESET=$2"); ENV_DISPLAY+=("STAGE1_EXP_PRESET=$2"); shift 2 ;;
    --session) SESSION_NAME="$2"; shift 2 ;;
    --max-minutes) MAX_RUNTIME_MIN="$2"; shift 2 ;;
    --max-retries) MAX_RETRIES="$2"; shift 2 ;;
    --sync-interval) SYNC_INTERVAL_SEC="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; CUSTOM_OUT_DIR=1; shift 2 ;;
    --credentials) CRED_PATH="$2"; shift 2 ;;
    --no-stop-on-alarm) STOP_ON_ALARM=0; shift ;;
    --status) ACTION="status"; shift ;;
    --stop-session) [ $# -ge 2 ] || { printf '[ERROR] --stop-session requires a session name\n' >&2; exit 2; }; ACTION="stop_session"; TARGET_STOP_SESSION="$2"; shift 2 ;;
    --stop-owned|--stop-all) ACTION="stop_owned"; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf '[ERROR] Unknown option: %s\n' "$1" >&2; usage; exit 2 ;;
  esac
done

for _g in "$GPU_TYPE" $(printf '%s' "$GPU_FALLBACK" | tr ',' ' '); do
  [ -n "$_g" ] || continue
  case "$_g" in T4|L4|G4|A100|H100) ;; *) printf '[ERROR] Unsupported GPU tier: %s\n' "$_g" >&2; exit 2 ;; esac
done

resolve_tier_hmem() {
  local tier="$1"
  if [ "$HIGH_MEM" = "1" ]; then [ "$tier" = "L4" ] && echo 0 || echo 1
  elif [ "$HIGH_MEM" = "0" ]; then echo 0
  else case "$tier" in H100|G4) echo 1 ;; *) echo 0 ;; esac
  fi
}

TIER_LIST="$GPU_TYPE"
for _t in $(printf '%s' "$GPU_FALLBACK" | tr ',' ' '); do
  [ -n "$_t" ] || continue
  case ",$TIER_LIST," in *,"$_t",*) ;; *) TIER_LIST="$TIER_LIST,$_t" ;; esac
done

if [ -z "$COLAB_BIN" ]; then
  if [ -x "$VENV_DIR/bin/colab" ]; then COLAB_BIN="$VENV_DIR/bin/colab"
  elif command -v colab >/dev/null 2>&1; then COLAB_BIN="$(command -v colab)"
  else COLAB_BIN="colab"; fi
fi

guard_cmd() { "$PY_BIN" "$SCRIPT_DIR/session_guard.py" "$@"; }
colab_global_cmd() { local rc="${COLAB_GUARD_STATE_DIR:-$SCRIPT_DIR/.state}/.readonly_sessions.json"; mkdir -p "$(dirname "$rc")"; GOOGLE_APPLICATION_CREDENTIALS="$CRED_PATH" "$COLAB_BIN" --config "$rc" --auth adc "$@"; }
[ "$ACTION" = "stop_session" ] && [ -z "$SESSION_NAME" ] && SESSION_NAME="$TARGET_STOP_SESSION"

if [ "$NOTEBOOK_KEY" = "nb1" ]; then
  : "${SESSION_NAME:=nb1}"; : "${MAX_RUNTIME_MIN:=210}"
  [ -z "$NOTEBOOK_FILE" ] && NOTEBOOK_FILE="$( [ "$GPU_TYPE" = "T4" ] && echo notebooks/01_Train_Stage1_Classifier_T4.ipynb || echo notebooks/01_Train_Stage1_Classifier_L4.ipynb )"
  CKPT_LATEST_NAME="stage1_latest.pth"; CKPT_BEST_NAME="stage1_retrained_best.pth"; CKPT_CLS_NAME="stage1_best_cls.pth"
  REMOTE_ZIP="/content/SE374_Stage1_Training_Results.zip"; REMOTE_METRICS_JSON="/content/stage1_results/STAGE1_TRAINING_METRICS.json"; REMOTE_PLOT_PNG="/content/stage1_results/stage1_training_curves.png"
elif [ "$NOTEBOOK_KEY" = "nb2" ]; then
  : "${SESSION_NAME:=nb2}"; : "${MAX_RUNTIME_MIN:=240}"
  [ -z "$NOTEBOOK_FILE" ] && NOTEBOOK_FILE="$( [ "$GPU_TYPE" = "T4" ] && echo notebooks/02_Pseudo_Masks_Train_Stage2_T4.ipynb || echo notebooks/02_Pseudo_Masks_Train_Stage2_L4.ipynb )"
  CKPT_LATEST_NAME="stage2_latest.pth"; CKPT_BEST_NAME="stage2_retrained_best.pth"; CKPT_CLS_NAME=""
  REMOTE_ZIP="/content/SE374_Stage2_Training_Results.zip"; REMOTE_METRICS_JSON="/content/stage2_results/STAGE2_TRAINING_METRICS.json"; REMOTE_PLOT_PNG="/content/stage2_results/stage2_training_loss.png"
else
  printf '[ERROR] Unsupported --notebook value: %s (expected nb1 or nb2)\n' "$NOTEBOOK_KEY" >&2; exit 2
fi

: "${OUTPUT_DIR:=vps_colab_runner/runs/$SESSION_NAME}"
CKPT_LOCAL_DIR="$OUTPUT_DIR/checkpoints"; ARTIFACTS_DIR="$OUTPUT_DIR/artifacts"
RUN_LOG="$OUTPUT_DIR/run.log"; STATE_DIR="$OUTPUT_DIR/.state"
SESS_COLAB_CFG="$STATE_DIR/colab_sessions.json"; EXEC_NB_COPY="$STATE_DIR/run_notebook.ipynb"
RESOLVED_HMEM="$(resolve_tier_hmem "$GPU_TYPE")"

colab_session_cmd() { GOOGLE_APPLICATION_CREDENTIALS="$CRED_PATH" "$COLAB_BIN" --config "$SESS_COLAB_CFG" --auth adc "$@"; }
log() {
  local msg="[VPS-RUNNER $(date '+%Y-%m-%d %H:%M:%S')] $*"
  printf '%s\n' "$msg"
  [ -d "$OUTPUT_DIR" ] && printf '%s\n' "$msg" >> "$RUN_LOG" 2>/dev/null || true
}
write_meta_env() {
  printf 'START_EPOCH=%s\nDEADLINE_EPOCH=%s\nMAX_MINUTES=%s\nNOTEBOOK=%s\nGPU=%s\nHIGH_MEM=%s\nASSIGNED_GPU=%s\nASSIGNED_HMEM=%s\nPRESET=%s\nSESSION=%s\nMAX_ATTEMPTS=%s\n' \
    "$START_EPOCH" "$DEADLINE_EPOCH" "$MAX_RUNTIME_MIN" "$NOTEBOOK_FILE" "$1" "$2" "$1" "$2" "${PRESET_NAME:-default}" "$SESSION_NAME" "$((MAX_RETRIES + 1))" > "$STATE_DIR/meta.env"
}

if [ "$ACTION" = "status" ]; then
  printf '=== Colab Usage & Sessions ===\n'
  colab_global_cmd usage || true; colab_global_cmd sessions || true
  printf '=== Local VPS Checkpoints (%s) ===\n' "$CKPT_LOCAL_DIR"
  ls -lh "$CKPT_LOCAL_DIR" 2>/dev/null || printf '(none yet)\n'; exit 0
fi
if [ "$ACTION" = "stop_session" ]; then
  guard_cmd validate-session --session "$TARGET_STOP_SESSION"
  log "Stopping owned Colab session '$TARGET_STOP_SESSION' (dual-match verified)..."
  STOP_CFG_ARGS=(); [ "$CUSTOM_OUT_DIR" -eq 1 ] && STOP_CFG_ARGS=(--colab-config "$SESS_COLAB_CFG")
  guard_cmd stop-session --session "$TARGET_STOP_SESSION" --cred-path "$CRED_PATH" --colab-bin "$COLAB_BIN" "${STOP_CFG_ARGS[@]}"
  colab_global_cmd usage || true; exit 0
fi
if [ "$ACTION" = "stop_owned" ]; then
  log "Stopping ALL active owned Colab sessions (foreign sessions untouched)..."
  guard_cmd stop-owned --cred-path "$CRED_PATH" --colab-bin "$COLAB_BIN"
  colab_global_cmd usage || true; exit 0
fi

guard_cmd validate-session --session "$SESSION_NAME"

if [ "$DRY_RUN" -eq 1 ]; then
  printf '[DRY-RUN] Notebook preset : %s (%s)\n' "$NOTEBOOK_KEY" "$NOTEBOOK_FILE"
  printf '[DRY-RUN] GPU / High-Mem  : %s / %s (mode=%s, tiers=%s)\n' "$GPU_TYPE" "$RESOLVED_HMEM" "$HIGH_MEM" "$TIER_LIST"
  printf '[DRY-RUN] Session / Preset: %s / %s | Output: %s\n' "$SESSION_NAME" "${PRESET_NAME:-default}" "$OUTPUT_DIR"
  printf '[DRY-RUN] Hard Wall-Clock : %s min | Max Retries: %s | Sync: %ss | StopOnAlarm: %s\n' "$MAX_RUNTIME_MIN" "$MAX_RETRIES" "$SYNC_INTERVAL_SEC" "$STOP_ON_ALARM"
  printf '[DRY-RUN] Config / Exec   : %s | %s\n' "$SESS_COLAB_CFG" "$EXEC_NB_COPY"
  printf '[DRY-RUN] Env / Creds     : %s | %s (exists=%s) | Colab: %s\n' "${ENV_DISPLAY[*]:-none}" "$CRED_PATH" "$([ -f "$CRED_PATH" ] && echo yes || echo no)" "$COLAB_BIN"
  [ -f "$NOTEBOOK_FILE" ] || { printf '[DRY-RUN ERROR] Notebook file not found: %s\n' "$NOTEBOOK_FILE" >&2; exit 1; }
  printf '[DRY-RUN] Validation PASS. Zero Colab sessions created.\n'; exit 0
fi

[ -f "$NOTEBOOK_FILE" ] || { printf '[ERROR] Notebook file not found: %s\n' "$NOTEBOOK_FILE" >&2; exit 1; }
[ -f "$CRED_PATH" ] || { printf '[ERROR] ADC credentials file not found: %s\n' "$CRED_PATH" >&2; exit 1; }
chmod 600 "$CRED_PATH" 2>/dev/null || true

mkdir -p "$CKPT_LOCAL_DIR" "$ARTIFACTS_DIR" "$STATE_DIR"
rm -f "$STATE_DIR/stop_reason" "$STATE_DIR/exec.pid"

START_EPOCH="$(date +%s)"; DEADLINE_EPOCH="$((START_EPOCH + MAX_RUNTIME_MIN * 60))"
write_meta_env "$GPU_TYPE" "$RESOLVED_HMEM"
PROV_LOCK="$(guard_cmd provision-lock-path --cred-path "$CRED_PATH")"
WATCHER_PID=""; EXEC_PID=""; TAIL_PID=""; CLEANED_UP=0; SESSION_OWNED_BY_PID=0; PROVISIONING_IN_PROGRESS=0

sync_remote_file_atomic() {
  local remote_path="$1" local_path="$2" mtime_state="$STATE_DIR/$(basename "$local_path").mtime" remote_meta remote_sha tmp_file local_sha
  remote_meta="$(colab_session_cmd ssh -s "$SESSION_NAME" -- stat -c '%Y %s' "$remote_path" 2>/dev/null || true)"
  [ -n "$remote_meta" ] || return 1
  if [ -f "$mtime_state" ] && [ -f "$local_path" ] && [ "$remote_meta" = "$(cat "$mtime_state" 2>/dev/null || true)" ]; then return 0; fi
  remote_sha="$(colab_session_cmd ssh -s "$SESSION_NAME" -- sha256sum "$remote_path" 2>/dev/null | awk '{print $1}' || true)"
  [ -n "$remote_sha" ] || return 1
  tmp_file="${local_path}.tmp.${BASHPID:-$$}"
  log "Syncing $remote_path -> $local_path (meta: $remote_meta)..."
  if colab_session_cmd ssh -s "$SESSION_NAME" -- cat "$remote_path" > "$tmp_file"; then
    local_sha="$(sha256sum "$tmp_file" | awk '{print $1}')"
    if [ "$local_sha" = "$remote_sha" ]; then
      mv -f "$tmp_file" "$local_path"; printf '%s' "$remote_meta" > "$mtime_state"
      log "Verified atomic sync OK: $local_path (SHA256=$local_sha)"; return 0
    fi
    log "WARN: SHA256 mismatch syncing $remote_path (expected=$remote_sha got=$local_sha)."
  fi
  rm -f "$tmp_file"; return 1
}

upload_checkpoint_for_resume() {
  local local_path="$1" remote_path="$2" local_sha remote_sha try
  [ -f "$local_path" ] || return 0
  [[ "$remote_path" =~ ^/[A-Za-z0-9_./-]+$ ]] || { log "ERROR: Unsafe remote_path: $remote_path"; return 1; }
  local_sha="$(sha256sum "$local_path" | awk '{print $1}')"
  log "Uploading local checkpoint $local_path -> $remote_path for auto-resume..."
  for try in 1 2 3; do
    if colab_session_cmd ssh -s "$SESSION_NAME" -- mkdir -p /content/checkpoints \
       && cat "$local_path" | colab_session_cmd ssh -s "$SESSION_NAME" -- sh -c "cat > '${remote_path}.tmp' && mv -f '${remote_path}.tmp' '${remote_path}'"; then
      remote_sha="$(colab_session_cmd ssh -s "$SESSION_NAME" -- sha256sum "$remote_path" 2>/dev/null | awk '{print $1}' || true)"
      [ "$local_sha" = "$remote_sha" ] && { log "Verified resume checkpoint uploaded: $remote_path (SHA256=$remote_sha)"; return 0; }
      log "WARN: Upload SHA256 mismatch (try $try/3) for $remote_path"
    else
      log "WARN: Resume checkpoint SSH upload failed (try $try/3) for $remote_path"
    fi
    [ "$try" -lt 3 ] && sleep 5
  done
  log "ERROR: Upload failed after 3 attempts for $remote_path"; return 1
}

sync_all_checkpoints() {
  sync_remote_file_atomic "/content/checkpoints/$CKPT_BEST_NAME" "$CKPT_LOCAL_DIR/$CKPT_BEST_NAME" || true
  sync_remote_file_atomic "/content/checkpoints/$CKPT_LATEST_NAME" "$CKPT_LOCAL_DIR/$CKPT_LATEST_NAME" || true
  [ -n "$CKPT_CLS_NAME" ] && sync_remote_file_atomic "/content/checkpoints/$CKPT_CLS_NAME" "$CKPT_LOCAL_DIR/$CKPT_CLS_NAME" || true
  sync_remote_file_atomic "$REMOTE_METRICS_JSON" "$ARTIFACTS_DIR/$(basename "$REMOTE_METRICS_JSON")" || true
}

download_final_artifacts() {
  log "Downloading final deliverables from session '$SESSION_NAME'..."
  sync_all_checkpoints
  sync_remote_file_atomic "$REMOTE_PLOT_PNG" "$ARTIFACTS_DIR/$(basename "$REMOTE_PLOT_PNG")" || true
  sync_remote_file_atomic "$REMOTE_ZIP" "$ARTIFACTS_DIR/$(basename "$REMOTE_ZIP")" || true
  colab_session_cmd log -s "$SESSION_NAME" -o "$ARTIFACTS_DIR/${SESSION_NAME}_session_log.md" || true
  local nb_out="${EXEC_NB_COPY%.ipynb}_output.ipynb"
  [ -f "$nb_out" ] && cp -f "$nb_out" "$ARTIFACTS_DIR/$(basename "$nb_out")" || true
}

trigger_stop_exec() {
  printf '%s' "$1" > "$STATE_DIR/stop_reason"; sync_all_checkpoints
  [ -f "$STATE_DIR/exec.pid" ] && kill -TERM "$(cat "$STATE_DIR/exec.pid")" 2>/dev/null || true
}

start_watcher() {
  local start_line="${1:-1}"
  (
    local last_sync now; last_sync="$(date +%s)"
    while true; do
      sleep 15; now="$(date +%s)"
      if [ "$now" -ge "$DEADLINE_EPOCH" ]; then
        log "WATCHDOG TIMEOUT: Reached ${MAX_RUNTIME_MIN} min limit. Stopping VM!"; trigger_stop_exec "WATCHDOG_TIMEOUT"; exit 0
      fi
      if [ -f "$RUN_LOG" ]; then
        if tail -n +"$start_line" "$RUN_LOG" | grep -E 'Loss: nan|Non-finite epoch loss|Train Loss=nan' >/dev/null; then
          log "FAIL-FAST ALARM: Non-finite/NaN loss detected in run.log!"; trigger_stop_exec "NAN_ALARM"; exit 0
        fi
        if [ "$STOP_ON_ALARM" -eq 1 ] && tail -n +"$start_line" "$RUN_LOG" | grep -E '\[EARLY REGRESSION ALARM\]|\[CALIBRATION COLLAPSE ALARM\]' >/dev/null; then
          log "FAIL-FAST ALARM: Calibration collapse / regression alarm triggered!"; trigger_stop_exec "CALIBRATION_COLLAPSE_ALARM"; exit 0
        fi
      fi
      if [ $((now - last_sync)) -ge "$SYNC_INTERVAL_SEC" ]; then last_sync="$now"; sync_all_checkpoints; fi
    done
  ) &
  WATCHER_PID="$!"
}

stop_bg() { [ -n "$1" ] && kill -0 "$1" 2>/dev/null && { kill -TERM "$1" 2>/dev/null || true; wait "$1" 2>/dev/null || true; } || true; }

stop_this_owned_session() {
  [ "${SESSION_OWNED_BY_PID:-0}" -eq 1 ] || return 0
  if guard_cmd stop-session --session "$SESSION_NAME" --colab-config "$SESS_COLAB_CFG" --cred-path "$CRED_PATH" --colab-bin "$COLAB_BIN"; then
    SESSION_OWNED_BY_PID=0; return 0
  fi
  log "WARN: stop-session failed for '$SESSION_NAME'; retrying once in 5s..."; sleep 5
  if guard_cmd stop-session --session "$SESSION_NAME" --colab-config "$SESS_COLAB_CFG" --cred-path "$CRED_PATH" --colab-bin "$COLAB_BIN"; then
    SESSION_OWNED_BY_PID=0; return 0
  fi
  log "ERROR: stop-session failed twice for '$SESSION_NAME'; keeping SESSION_OWNED_BY_PID=1."; return 1
}

release_prov_lock() { flock -u 9 2>/dev/null || true; exec 9>&- 2>/dev/null || true; }

cleanup() {
  [ "$CLEANED_UP" -eq 1 ] && return 0
  CLEANED_UP=1
  stop_bg "$WATCHER_PID"; stop_bg "$EXEC_PID"; stop_bg "$TAIL_PID"
  if [ "${SESSION_OWNED_BY_PID:-0}" -eq 1 ]; then
    log "CLEANUP GUARD: Stopping verified owned session '$SESSION_NAME'..."; stop_this_owned_session || true
  elif [ "${PROVISIONING_IN_PROGRESS:-0}" -eq 1 ]; then
    log "CLEANUP GUARD: Cleaning up partial provision for '$SESSION_NAME'..."
    guard_cmd cleanup-partial --session "$SESSION_NAME" --colab-config "$SESS_COLAB_CFG" --cred-path "$CRED_PATH" --colab-bin "$COLAB_BIN" || true
    PROVISIONING_IN_PROGRESS=0
  else
    log "CLEANUP GUARD: Session '$SESSION_NAME' not owned by this PID; skipping Colab stop."
  fi
  release_prov_lock; log "Post-teardown Colab usage check:"; colab_global_cmd usage || true
}

trap cleanup EXIT INT TERM

log "Starting resilient Colab run: nb=$NOTEBOOK_FILE tiers=$TIER_LIST hmem_mode=$HIGH_MEM session=$SESSION_NAME preset=${PRESET_NAME:-default} max_min=$MAX_RUNTIME_MIN"
colab_global_cmd usage || true

ATTEMPT=1; MAX_ATTEMPTS="$((MAX_RETRIES + 1))"; SUCCEEDED=0

while [ "$ATTEMPT" -le "$MAX_ATTEMPTS" ]; do
  [ "$(date +%s)" -ge "$DEADLINE_EPOCH" ] && { log "Hard wall-clock limit reached before attempt $ATTEMPT."; break; }
  printf '%s' "$ATTEMPT" > "$STATE_DIR/attempt"
  stop_this_owned_session || { log "ERROR: Prior owned session '$SESSION_NAME' could not be stopped; aborting."; break; }

  mkdir -p "$(dirname "$PROV_LOCK")"; exec 9>"$PROV_LOCK"; flock -x 9
  guard_cmd snapshot-protected --cred-path "$CRED_PATH" --colab-bin "$COLAB_BIN" >/dev/null
  guard_cmd assert-can-create --session "$SESSION_NAME" --cred-path "$CRED_PATH" --colab-config "$SESS_COLAB_CFG"
  guard_cmd cleanup-partial --session "$SESSION_NAME" --colab-config "$SESS_COLAB_CFG" --cred-path "$CRED_PATH" --colab-bin "$COLAB_BIN" >/dev/null

  PROVISIONED=0; ASSIGNED_GPU="$GPU_TYPE"; ASSIGNED_HMEM="$RESOLVED_HMEM"
  for TRY_GPU in $(printf '%s' "$TIER_LIST" | tr ',' ' '); do
    TRY_HMEM="$(resolve_tier_hmem "$TRY_GPU")"; HMEM_ARGS=(); [ "$TRY_HMEM" -eq 1 ] && HMEM_ARGS=(--high-mem)
    log "=== Attempt $ATTEMPT/$MAX_ATTEMPTS: Provisioning owned session '$SESSION_NAME' ($TRY_GPU, high_mem=$TRY_HMEM) ==="
    PROVISIONING_IN_PROGRESS=1
    if colab_session_cmd new -s "$SESSION_NAME" --gpu "$TRY_GPU" "${HMEM_ARGS[@]}"; then
      if guard_cmd register-owned --session "$SESSION_NAME" --colab-config "$SESS_COLAB_CFG" --cred-path "$CRED_PATH" --gpu "$TRY_GPU" --high-mem "$TRY_HMEM" --pid "$$" >/dev/null; then
        SESSION_OWNED_BY_PID=1; PROVISIONING_IN_PROGRESS=0; PROVISIONED=1; ASSIGNED_GPU="$TRY_GPU"; ASSIGNED_HMEM="$TRY_HMEM"; break
      fi
      log "ERROR: register-owned failed after colab new for '$SESSION_NAME'; cleaning up partial endpoint..."
      guard_cmd cleanup-partial --session "$SESSION_NAME" --colab-config "$SESS_COLAB_CFG" --cred-path "$CRED_PATH" --colab-bin "$COLAB_BIN" || true
      PROVISIONING_IN_PROGRESS=0; break
    fi
    log "WARN: Failed to provision GPU tier '$TRY_GPU' for '$SESSION_NAME'."
    guard_cmd cleanup-partial --session "$SESSION_NAME" --colab-config "$SESS_COLAB_CFG" --cred-path "$CRED_PATH" --colab-bin "$COLAB_BIN" || true
    PROVISIONING_IN_PROGRESS=0
  done
  release_prov_lock
  [ "$PROVISIONED" -eq 1 ] || { log "ERROR: All GPU tiers failed ($TIER_LIST)."; break; }
  write_meta_env "$ASSIGNED_GPU" "$ASSIGNED_HMEM"

  STAGE1_UP_SRC=""
  if [ "$NOTEBOOK_KEY" = "nb2" ]; then
    if [ -n "${STAGE1_LOCAL_CKPT:-}" ] && [ -f "$STAGE1_LOCAL_CKPT" ]; then STAGE1_UP_SRC="$STAGE1_LOCAL_CKPT"
    elif [ -f "$CKPT_LOCAL_DIR/stage1_retrained_best.pth" ]; then STAGE1_UP_SRC="$CKPT_LOCAL_DIR/stage1_retrained_best.pth"; fi
  fi
  if [ -f "$CKPT_LOCAL_DIR/$CKPT_LATEST_NAME" ] || [ -f "$CKPT_LOCAL_DIR/$CKPT_BEST_NAME" ] || { [ -n "$CKPT_CLS_NAME" ] && [ -f "$CKPT_LOCAL_DIR/$CKPT_CLS_NAME" ]; } || [ -n "$STAGE1_UP_SRC" ]; then
    log "Found saved checkpoints on VPS disk. Uploading to '$SESSION_NAME' for automatic resume..."
    if ! upload_checkpoint_for_resume "$CKPT_LOCAL_DIR/$CKPT_LATEST_NAME" "/content/checkpoints/$CKPT_LATEST_NAME" \
       || ! upload_checkpoint_for_resume "$CKPT_LOCAL_DIR/$CKPT_BEST_NAME" "/content/checkpoints/$CKPT_BEST_NAME" \
       || { [ -n "$CKPT_CLS_NAME" ] && ! upload_checkpoint_for_resume "$CKPT_LOCAL_DIR/$CKPT_CLS_NAME" "/content/checkpoints/$CKPT_CLS_NAME"; } \
       || { [ -n "$STAGE1_UP_SRC" ] && ! upload_checkpoint_for_resume "$STAGE1_UP_SRC" "/content/checkpoints/stage1_retrained_best.pth"; }; then
      log "WARN: Resume checkpoint upload failed; stopping attempt $ATTEMPT and retrying..."
      stop_this_owned_session || { log "ERROR: Could not stop owned session after upload failure."; break; }
      [ "$ATTEMPT" -lt "$MAX_ATTEMPTS" ] && sleep 15
      ATTEMPT="$((ATTEMPT + 1))"; continue
    fi
  fi

  cp -f "$NOTEBOOK_FILE" "$EXEC_NB_COPY"; touch "$RUN_LOG"
  LOG_START_LINE="$(( $(wc -l < "$RUN_LOG" 2>/dev/null || echo 0) + 1 ))"
  start_watcher "$LOG_START_LINE"; tail -n 0 -f "$RUN_LOG" &
  TAIL_PID="$!"
  set +e
  colab_session_cmd exec -s "$SESSION_NAME" -f "$EXEC_NB_COPY" --timeout 0 "${ENV_ARGS[@]}" >> "$RUN_LOG" 2>&1 &
  EXEC_PID="$!"; printf '%s' "$EXEC_PID" > "$STATE_DIR/exec.pid"
  wait "$EXEC_PID"; EXEC_RC=$?
  set -e
  EXEC_PID=""; rm -f "$STATE_DIR/exec.pid"
  stop_bg "$TAIL_PID"; TAIL_PID=""; stop_bg "$WATCHER_PID"; WATCHER_PID=""
  sync_all_checkpoints

  if [ -f "$STATE_DIR/stop_reason" ]; then log "Halting loop due to safety guard: $(cat "$STATE_DIR/stop_reason")"; break; fi
  if [ "$EXEC_RC" -eq 0 ] && [ -f "${EXEC_NB_COPY%.ipynb}_output.ipynb" ]; then
    set +e
    NB_ERR_MSG="$("$PY_BIN" -c 'import json,re,sys;nb=json.load(open(sys.argv[1],encoding="utf-8"));errs=[(i,o.get("ename",""),o.get("evalue",""),"\n".join(o.get("traceback",[]))) for i,c in enumerate(nb.get("cells",[])) for o in c.get("outputs",[]) if o.get("output_type")=="error"];errs and (print("; ".join(f"cell_{i}: {a}: {b}" for i,a,b,_ in errs)[:400]),sys.exit(2 if any(re.search(r"STAGE1_GIT_REF|Non-finite epoch loss|CALIBRATION COLLAPSE|strict=True",f"{a} {b} {tb}") for _,a,b,tb in errs) else 1))' "${EXEC_NB_COPY%.ipynb}_output.ipynb" 2>&1)"
    NB_ERR_RC=$?
    set -e
    if [ "$NB_ERR_RC" -ne 0 ]; then
      log "[ERROR] Notebook cell exception: $NB_ERR_MSG"
      [ "$NB_ERR_RC" -eq 2 ] && { printf 'NOTEBOOK_CELL_ERROR' > "$STATE_DIR/stop_reason"; cp -f "${EXEC_NB_COPY%.ipynb}_output.ipynb" "$ARTIFACTS_DIR/" 2>/dev/null || true; break; }
      EXEC_RC=1
    fi
  fi
  if [ "$EXEC_RC" -eq 0 ]; then
    log "Notebook execution completed with exit code 0!"
    download_final_artifacts
    if tail -n +"$LOG_START_LINE" "$RUN_LOG" | grep -E '\[EARLY STOP( PATIENCE)?\]' >/dev/null; then
      printf 'EARLY_STOP_PATIENCE' > "$STATE_DIR/stop_reason"; log "Clean early stop patience recorded after final artifact sync."
    fi
    SUCCEEDED=1; break
  fi

  log "WARN: Attempt $ATTEMPT exited with code $EXEC_RC (possible disconnect). Salvaging..."
  stop_this_owned_session || { log "ERROR: Could not stop owned session '$SESSION_NAME'; aborting retries."; break; }
  [ "$ATTEMPT" -lt "$MAX_ATTEMPTS" ] && { log "Waiting 15s before auto-resume attempt $((ATTEMPT + 1))/$MAX_ATTEMPTS..."; sleep 15; }
  ATTEMPT="$((ATTEMPT + 1))"
done

if [ "$SUCCEEDED" -eq 1 ]; then
  log "SUCCESS: All deliverables saved in $OUTPUT_DIR. Cleanup trap will now terminate '$SESSION_NAME'."; exit 0
fi
log "STOPPED WITHOUT FULL COMPLETION. Latest salvaged checkpoints are in $CKPT_LOCAL_DIR."; exit 1
