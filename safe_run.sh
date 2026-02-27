#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'USAGE'
Usage:
  ./safe_run.sh [--timeout SEC] [--log-dir DIR] [--summary-json PATH] [--heartbeat-sec N] [--tail-lines N] [--no-echo] "<command string>"
  ./safe_run.sh [--timeout SEC] [--log-dir DIR] [--summary-json PATH] [--heartbeat-sec N] [--tail-lines N] [--no-echo] <command> [args...]

Examples:
  ./safe_run.sh --timeout 60 "git status --short"
  ./safe_run.sh --timeout 180 pytest -q
USAGE
}

timeout_sec=600
log_dir=".safe_run/logs"
show_output=1
summary_json=""
heartbeat_sec=30
tail_lines=100

while [[ $# -gt 0 ]]; do
  case "$1" in
    --timeout)
      [[ $# -ge 2 ]] || { echo "[safe_run] missing value for --timeout" >&2; exit 2; }
      timeout_sec="$2"
      shift 2
      ;;
    --log-dir)
      [[ $# -ge 2 ]] || { echo "[safe_run] missing value for --log-dir" >&2; exit 2; }
      log_dir="$2"
      shift 2
      ;;
    --summary-json)
      [[ $# -ge 2 ]] || { echo "[safe_run] missing value for --summary-json" >&2; exit 2; }
      summary_json="$2"
      shift 2
      ;;
    --heartbeat-sec)
      [[ $# -ge 2 ]] || { echo "[safe_run] missing value for --heartbeat-sec" >&2; exit 2; }
      heartbeat_sec="$2"
      shift 2
      ;;
    --tail-lines)
      [[ $# -ge 2 ]] || { echo "[safe_run] missing value for --tail-lines" >&2; exit 2; }
      tail_lines="$2"
      shift 2
      ;;
    --quiet)
      show_output=0
      shift
      ;;
    --no-echo)
      show_output=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      break
      ;;
  esac
done

[[ $# -gt 0 ]] || { usage >&2; exit 2; }

mkdir -p "$log_dir"
ts="$(date +%Y%m%d_%H%M%S)"
rand="$RANDOM"
run_id="${ts}_${rand}"
out_log="$log_dir/${run_id}.out.log"
err_log="$log_dir/${run_id}.err.log"
summary_path="${summary_json:-$log_dir/${run_id}.summary.json}"
mkdir -p "$(dirname "$summary_path")"
cwd_now="$(pwd)"

echo "[safe_run] cwd: $cwd_now"
echo "[safe_run] timeout: ${timeout_sec}s"

display_cmd=""
if [[ $# -eq 1 ]]; then
  cmd_str="$1"
  display_cmd="$cmd_str"
  cmd_mode="string"
  cmd_exe="bash"
else
  display_cmd="$*"
  cmd_mode="argv"
  cmd_exe="$1"
fi

echo "[safe_run] run_id: $run_id"
echo "[safe_run] cmd: $display_cmd"
echo "[safe_run] out_log: $out_log"
echo "[safe_run] err_log: $err_log"
echo "[safe_run] summary: $summary_path"

start_ts="$(date +%s)"
start_iso="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
timed_out=0
killed_tree=0
child_pid=""
started_in_group=0

if command -v setsid >/dev/null 2>&1; then
  started_in_group=1
fi

if [[ "$cmd_mode" == "string" ]]; then
  if [[ "$started_in_group" -eq 1 ]]; then
    # Start in a new process group so timeout cleanup can terminate descendants.
    setsid bash -lc "$cmd_str" >"$out_log" 2>"$err_log" &
  else
    bash -lc "$cmd_str" >"$out_log" 2>"$err_log" &
  fi
else
  if [[ "$started_in_group" -eq 1 ]]; then
    setsid "$@" >"$out_log" 2>"$err_log" &
  else
    "$@" >"$out_log" 2>"$err_log" &
  fi
fi
child_pid=$!

deadline=$((start_ts + timeout_sec))
next_hb=$((start_ts + heartbeat_sec))
while kill -0 "$child_pid" 2>/dev/null; do
  now="$(date +%s)"
  if [[ "$heartbeat_sec" -gt 0 && "$now" -ge "$next_hb" ]]; then
    elapsed=$((now - start_ts))
    echo "[safe_run] heartbeat: alive pid=$child_pid elapsed=${elapsed}s"
    next_hb=$((now + heartbeat_sec))
  fi
  if [[ "$now" -ge "$deadline" ]]; then
    timed_out=1
    if [[ "$started_in_group" -eq 1 ]]; then
      kill -TERM -- "-$child_pid" 2>/dev/null && killed_tree=1 || true
    else
      kill -TERM "$child_pid" 2>/dev/null && killed_tree=1 || true
    fi
    sleep 1
    if [[ "$started_in_group" -eq 1 ]]; then
      kill -KILL -- "-$child_pid" 2>/dev/null || true
    else
      kill -KILL "$child_pid" 2>/dev/null || true
    fi
    break
  fi
  sleep 0.25
done

set +e
wait "$child_pid"
raw_exit_code=$?
set -e

if [[ "$timed_out" -eq 1 ]]; then
  echo "[safe_run] TIMEOUT after ${timeout_sec}s" >>"$err_log"
  exit_code=124
else
  exit_code=$raw_exit_code
fi

end_ts="$(date +%s)"
end_iso="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
duration_sec="$((end_ts - start_ts))"

summary_writer=""
if command -v python3 >/dev/null 2>&1; then
  summary_writer="python3"
elif command -v python >/dev/null 2>&1; then
  summary_writer="python"
fi

if [[ -n "$summary_writer" ]]; then
  if [[ "$cmd_mode" == "string" ]]; then
    "$summary_writer" - <<'PY' "$summary_path" "$run_id" "$display_cmd" "$timeout_sec" "$heartbeat_sec" "$tail_lines" "$child_pid" "$start_iso" "$end_iso" "$duration_sec" "$exit_code" "$timed_out" "$killed_tree" "$out_log" "$err_log" "$cwd_now" "$cmd_mode" "$cmd_exe" "$cmd_str" || true
import json, sys
(
    summary_path,
    run_id,
    display_cmd,
    timeout_sec,
    heartbeat_sec,
    tail_lines,
    pid,
    start_iso,
    end_iso,
    duration_sec,
    exit_code,
    timed_out,
    killed_tree,
    out_log,
    err_log,
    cwd_value,
    cmd_mode,
    cmd_exe,
    cmd_str,
) = sys.argv[1:]
payload = {
    "schema": "safe_run_result_v1",
    "run_id": run_id,
    "generated_at": end_iso,
    "cwd": cwd_value,
    "command": {"exe": cmd_exe, "args": ["-lc", cmd_str], "display": display_cmd},
    "timeout_sec": int(timeout_sec),
    "heartbeat_sec": int(heartbeat_sec),
    "tail_lines": int(tail_lines),
    "pid": int(pid or 0),
    "start_at_utc": start_iso,
    "end_at_utc": end_iso,
    "duration_sec": float(duration_sec),
    "exit_code": int(exit_code),
    "timed_out": bool(int(timed_out)),
    "killed_process_tree": bool(int(killed_tree)),
    "stdout_log": out_log,
    "stderr_log": err_log,
}
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)
    f.write("\n")
PY
  else
    "$summary_writer" - <<'PY' "$summary_path" "$run_id" "$display_cmd" "$timeout_sec" "$heartbeat_sec" "$tail_lines" "$child_pid" "$start_iso" "$end_iso" "$duration_sec" "$exit_code" "$timed_out" "$killed_tree" "$out_log" "$err_log" "$cwd_now" "$cmd_mode" "$cmd_exe" "${@:2}" || true
import json, sys
(
    summary_path,
    run_id,
    display_cmd,
    timeout_sec,
    heartbeat_sec,
    tail_lines,
    pid,
    start_iso,
    end_iso,
    duration_sec,
    exit_code,
    timed_out,
    killed_tree,
    out_log,
    err_log,
    cwd_value,
    cmd_mode,
    cmd_exe,
    *cmd_args,
) = sys.argv[1:]
payload = {
    "schema": "safe_run_result_v1",
    "run_id": run_id,
    "generated_at": end_iso,
    "cwd": cwd_value,
    "command": {"exe": cmd_exe, "args": cmd_args, "display": display_cmd},
    "timeout_sec": int(timeout_sec),
    "heartbeat_sec": int(heartbeat_sec),
    "tail_lines": int(tail_lines),
    "pid": int(pid or 0),
    "start_at_utc": start_iso,
    "end_at_utc": end_iso,
    "duration_sec": float(duration_sec),
    "exit_code": int(exit_code),
    "timed_out": bool(int(timed_out)),
    "killed_process_tree": bool(int(killed_tree)),
    "stdout_log": out_log,
    "stderr_log": err_log,
}
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)
    f.write("\n")
PY
  fi
fi

echo "[safe_run] exit_code: $exit_code"

if [[ $exit_code -ne 0 ]]; then
  echo "========== safe_run failure summary =========="
  echo "command: $display_cmd"
  echo "exit_code: $exit_code"
  echo "--- last ${tail_lines} lines (stdout) ---"
  tail -n "$tail_lines" "$out_log" || true
  echo "--- last ${tail_lines} lines (stderr) ---"
  tail -n "$tail_lines" "$err_log" || true
  echo "============================================="
  exit "$exit_code"
fi

if [[ $show_output -eq 1 ]]; then
  if [[ -s "$out_log" ]]; then
    cat "$out_log"
  fi
  if [[ -s "$err_log" ]]; then
    cat "$err_log" >&2
  fi
fi

exit 0
