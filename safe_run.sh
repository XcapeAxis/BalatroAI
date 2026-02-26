#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'USAGE'
Usage:
  ./safe_run.sh [--timeout SEC] [--log-dir DIR] "<command string>"
  ./safe_run.sh [--timeout SEC] [--log-dir DIR] <command> [args...]

Examples:
  ./safe_run.sh --timeout 60 "git status --short"
  ./safe_run.sh --timeout 180 pytest -q
USAGE
}

timeout_sec=600
log_dir=".safe_run/logs"
show_output=1

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
    --quiet)
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
out_log="$log_dir/${ts}_${rand}.out.log"
err_log="$log_dir/${ts}_${rand}.err.log"

echo "[safe_run] cwd: $(pwd)"
echo "[safe_run] timeout: ${timeout_sec}s"

if [[ $# -eq 1 ]]; then
  cmd_str="$1"
  display_cmd="$cmd_str"
  echo "[safe_run] cmd: $display_cmd"
  echo "[safe_run] out_log: $out_log"
  echo "[safe_run] err_log: $err_log"
  set +e
  timeout --preserve-status "${timeout_sec}s" bash -lc "$cmd_str" >"$out_log" 2>"$err_log"
  exit_code=$?
  set -e
else
  display_cmd="$*"
  echo "[safe_run] cmd: $display_cmd"
  echo "[safe_run] out_log: $out_log"
  echo "[safe_run] err_log: $err_log"
  set +e
  timeout --preserve-status "${timeout_sec}s" "$@" >"$out_log" 2>"$err_log"
  exit_code=$?
  set -e
fi

echo "[safe_run] exit_code: $exit_code"

if [[ $exit_code -ne 0 ]]; then
  echo "========== safe_run failure summary =========="
  echo "command: $display_cmd"
  echo "exit_code: $exit_code"
  echo "--- last 100 lines (stdout) ---"
  tail -n 100 "$out_log" || true
  echo "--- last 100 lines (stderr) ---"
  tail -n 100 "$err_log" || true
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
