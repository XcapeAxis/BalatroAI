import argparse
import csv
import multiprocessing
import os
import platform
import re
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

SCALING_SUMMARY_RE = re.compile(
    r"Scaling summary:\s*instances=\d+.*?steps/s_total=([0-9.]+)",
    re.MULTILINE,
)
STEPS_RE = re.compile(r"^Steps:\s*(\d+)\s*$", re.MULTILINE)
BENCHMARK_WALL_RE = re.compile(r"^Benchmark wall time:\s*([0-9.]+)\s*sec\s*$", re.MULTILINE)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run throughput sweep against benchmark_balatrobot.py across multiple instance counts."
    )
    parser.add_argument("--benchmark", default="benchmark_balatrobot.py", help="Path to benchmark script.")
    parser.add_argument("--python", default=sys.executable, help="Python executable used to run benchmark.")
    parser.add_argument(
        "--instances",
        default="1,2,3,4,5,6,8",
        help="Comma-separated instance counts to sweep.",
    )
    parser.add_argument("--repeats", type=int, default=3, help="Repeat runs per sweep point.")
    parser.add_argument(
        "--mode",
        choices=["action_only", "rl_step", "both"],
        default="action_only",
        help="Benchmark mode sweep selection.",
    )
    parser.add_argument("--steps-per-instance", type=int, default=200, help="Per-instance steps for each run.")
    parser.add_argument(
        "--launch-instances",
        dest="launch_instances",
        action="store_true",
        help="Pass --launch-instances to benchmark.",
    )
    parser.add_argument(
        "--no-launch-instances",
        dest="launch_instances",
        action="store_false",
        help="Do not pass --launch-instances to benchmark.",
    )
    parser.set_defaults(launch_instances=True)
    parser.add_argument(
        "--launcher",
        choices=["direct", "uvx"],
        default="direct",
        help="Launcher backend passed to benchmark.",
    )
    parser.add_argument(
        "--balatrobot-cmd",
        default="balatrobot",
        help="balatrobot command passed in direct launcher mode.",
    )
    parser.add_argument(
        "--uvx-path",
        default="uvx",
        help="uvx command passed in uvx launcher mode.",
    )
    parser.add_argument("--love-path", default=None, help="Balatro executable path passed to benchmark.")
    parser.add_argument("--lovely-path", default=None, help="Lovely DLL path passed to benchmark.")
    parser.add_argument("--base-port", type=int, default=12346, help="Base port for the first run.")
    parser.add_argument(
        "--port-stride",
        type=int,
        default=50,
        help="Port offset between two consecutive runs.",
    )
    parser.add_argument("--stagger-start", type=float, default=0.2, help="Stagger start passed to benchmark.")
    parser.add_argument("--timeout", type=int, default=600, help="Per-run timeout (seconds).")
    parser.add_argument("--csv-out", default=None, help="Optional detailed CSV output path.")
    parser.add_argument(
        "--raw-dir",
        default=None,
        help="Optional directory for raw stdout/stderr dumps per run.",
    )
    return parser.parse_args()


def parse_instances(csv_text):
    values = []
    seen = set()
    for piece in csv_text.split(","):
        token = piece.strip()
        if not token:
            continue
        try:
            value = int(token)
        except ValueError as exc:
            raise ValueError(f"Invalid instance value: {token}") from exc
        if value <= 0:
            raise ValueError(f"Instance value must be > 0: {value}")
        if value in seen:
            continue
        seen.add(value)
        values.append(value)
    if not values:
        raise ValueError("No valid instance values provided.")
    return values


def resolve_modes(mode_arg):
    if mode_arg == "both":
        return ["action_only", "rl_step"]
    return [mode_arg]


def build_ports(base_port, instances, run_index, port_stride):
    base_port_used = base_port + run_index * port_stride
    ports = [str(base_port_used + i) for i in range(instances)]
    return base_port_used, ",".join(ports)


def build_cmd(args, mode, instances, ports_csv):
    cmd = [
        args.python,
        args.benchmark,
        "--instances",
        str(instances),
        "--mode",
        mode,
        "--steps-per-instance",
        str(args.steps_per_instance),
        "--ports",
        ports_csv,
        "--stagger-start",
        str(args.stagger_start),
        "--launcher",
        args.launcher,
    ]

    if args.launch_instances:
        cmd.append("--launch-instances")

    if args.launcher == "direct":
        cmd.extend(["--balatrobot-cmd", args.balatrobot_cmd])
    else:
        cmd.extend(["--uvx-path", args.uvx_path])

    if args.love_path:
        cmd.extend(["--love-path", args.love_path])
    if args.lovely_path:
        cmd.extend(["--lovely-path", args.lovely_path])

    return cmd


def ensure_text(value):
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return str(value)


def save_raw_output(raw_dir, run_meta, cmd, returncode, stdout_text, stderr_text):
    if raw_dir is None:
        return None

    raw_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    filename = (
        f"{ts}_mode-{run_meta['mode']}_inst-{run_meta['instances']}"
        f"_rep-{run_meta['repeat']}_run-{run_meta['run_index']}.txt"
    )
    path = raw_dir / filename
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(f"timestamp={datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"mode={run_meta['mode']}\n")
        f.write(f"instances={run_meta['instances']}\n")
        f.write(f"repeat={run_meta['repeat']}\n")
        f.write(f"run_index={run_meta['run_index']}\n")
        f.write(f"base_port_used={run_meta['base_port_used']}\n")
        f.write(f"returncode={returncode}\n")
        f.write(f"cmd={subprocess.list2cmdline(cmd)}\n")
        f.write("\n[stdout]\n")
        f.write(stdout_text)
        f.write("\n\n[stderr]\n")
        f.write(stderr_text)
        f.write("\n")
    return path


def run_once(cmd, timeout_sec, raw_dir, run_meta):
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=timeout_sec,
            check=False,
        )
        stdout_text = ensure_text(completed.stdout)
        stderr_text = ensure_text(completed.stderr)
        raw_path = save_raw_output(
            raw_dir=raw_dir,
            run_meta=run_meta,
            cmd=cmd,
            returncode=completed.returncode,
            stdout_text=stdout_text,
            stderr_text=stderr_text,
        )
        return {
            "timeout": False,
            "returncode": completed.returncode,
            "stdout": stdout_text,
            "stderr": stderr_text,
            "raw_path": raw_path,
        }
    except subprocess.TimeoutExpired as exc:
        stdout_text = ensure_text(exc.stdout)
        stderr_text = ensure_text(exc.stderr)
        raw_path = save_raw_output(
            raw_dir=raw_dir,
            run_meta=run_meta,
            cmd=cmd,
            returncode="TIMEOUT",
            stdout_text=stdout_text,
            stderr_text=stderr_text,
        )
        return {
            "timeout": True,
            "returncode": None,
            "stdout": stdout_text,
            "stderr": stderr_text,
            "raw_path": raw_path,
        }


def parse_steps_s_total(stdout_text, instances):
    summary_match = SCALING_SUMMARY_RE.search(stdout_text)
    if summary_match:
        return float(summary_match.group(1)), "scaling_summary"

    if instances == 1:
        steps_match = STEPS_RE.search(stdout_text)
        wall_match = BENCHMARK_WALL_RE.search(stdout_text)
        if steps_match and wall_match:
            steps = int(steps_match.group(1))
            wall = float(wall_match.group(1))
            if wall > 0:
                return steps / wall, "computed_from_steps_wall"

    raise ValueError("Could not parse steps/s_total from benchmark stdout.")


def parse_method_avg_ms(stdout_text, method_name):
    pattern = re.compile(
        rf"^\s*{re.escape(method_name)}:\s*count=\d+,\s*avg=([0-9.]+)ms",
        re.MULTILINE,
    )
    matches = pattern.findall(stdout_text)
    if not matches:
        return None
    return float(matches[-1])


def summarize_results(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["mode"], record["instances"])].append(record["steps_s_total"])

    summary_rows = []
    for (mode, instances), values in grouped.items():
        repeats = len(values)
        mean_val = statistics.mean(values)
        stdev_val = statistics.stdev(values) if repeats > 1 else 0.0
        summary_rows.append(
            {
                "mode": mode,
                "instances": instances,
                "repeats": repeats,
                "mean": mean_val,
                "stdev": stdev_val,
                "min": min(values),
                "max": max(values),
            }
        )
    return summary_rows


def print_summary_table(summary_rows, modes):
    print("\nSweep summary:")
    header = (
        f"{'mode':<12}"
        f"{'instances':>10}"
        f"{'repeats':>9}"
        f"{'mean_steps_s':>15}"
        f"{'stdev':>12}"
        f"{'min':>12}"
        f"{'max':>12}"
    )

    for mode in modes:
        mode_rows = [row for row in summary_rows if row["mode"] == mode]
        if not mode_rows:
            continue
        mode_rows.sort(key=lambda x: x["instances"])
        print(f"\nmode={mode}")
        print(header)
        for row in mode_rows:
            print(
                f"{row['mode']:<12}"
                f"{row['instances']:>10d}"
                f"{row['repeats']:>9d}"
                f"{row['mean']:>15.4f}"
                f"{row['stdev']:>12.4f}"
                f"{row['min']:>12.4f}"
                f"{row['max']:>12.4f}"
            )


def print_platform_info():
    print("\nPlatform:")
    print(f"  python={platform.python_version()} ({sys.executable})")
    print(f"  platform={platform.platform()}")
    print(f"  os_name={os.name}")
    print(f"  cpu_logical={multiprocessing.cpu_count()}")


def write_detail_csv(csv_path, records):
    fieldnames = [
        "timestamp",
        "mode",
        "instances",
        "repeat",
        "steps_s_total",
        "base_port_used",
        "returncode",
        "parse_source",
        "play_avg_ms",
        "gamestate_avg_ms",
        "cmd",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow(
                {
                    "timestamp": row["timestamp"],
                    "mode": row["mode"],
                    "instances": row["instances"],
                    "repeat": row["repeat"],
                    "steps_s_total": f"{row['steps_s_total']:.6f}",
                    "base_port_used": row["base_port_used"],
                    "returncode": row["returncode"],
                    "parse_source": row["parse_source"],
                    "play_avg_ms": "" if row["play_avg_ms"] is None else f"{row['play_avg_ms']:.6f}",
                    "gamestate_avg_ms": "" if row["gamestate_avg_ms"] is None else f"{row['gamestate_avg_ms']:.6f}",
                    "cmd": row["cmd"],
                }
            )


def write_summary_csv(summary_path, summary_rows):
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["mode", "instances", "repeats", "mean", "stdev", "min", "max"],
        )
        writer.writeheader()
        for row in sorted(summary_rows, key=lambda x: (x["mode"], x["instances"])):
            writer.writerow(
                {
                    "mode": row["mode"],
                    "instances": row["instances"],
                    "repeats": row["repeats"],
                    "mean": f"{row['mean']:.6f}",
                    "stdev": f"{row['stdev']:.6f}",
                    "min": f"{row['min']:.6f}",
                    "max": f"{row['max']:.6f}",
                }
            )


def validate_inputs(args, instances, modes):
    benchmark_path = Path(args.benchmark).expanduser()
    if not benchmark_path.is_absolute():
        benchmark_path = (Path.cwd() / benchmark_path).resolve()
    if not benchmark_path.exists():
        raise ValueError(f"Benchmark script not found: {benchmark_path}")
    if not benchmark_path.is_file():
        raise ValueError(f"Benchmark path is not a file: {benchmark_path}")

    if args.repeats <= 0:
        raise ValueError("--repeats must be > 0")
    if args.steps_per_instance <= 0:
        raise ValueError("--steps-per-instance must be > 0")
    if args.timeout <= 0:
        raise ValueError("--timeout must be > 0")
    if args.base_port <= 0 or args.base_port > 65535:
        raise ValueError("--base-port must be in 1..65535")
    if args.port_stride <= 0:
        raise ValueError("--port-stride must be > 0")
    if args.stagger_start < 0:
        raise ValueError("--stagger-start must be >= 0")
    if args.launcher == "direct" and not args.balatrobot_cmd:
        raise ValueError("--balatrobot-cmd cannot be empty in direct mode")
    if args.launcher == "uvx" and not args.uvx_path:
        raise ValueError("--uvx-path cannot be empty in uvx mode")

    total_runs = len(modes) * len(instances) * args.repeats
    max_port = args.base_port + (total_runs - 1) * args.port_stride + max(instances) - 1
    if max_port > 65535:
        raise ValueError(
            "Computed max port exceeds 65535. Lower --base-port, --port-stride, "
            "instances list, or total run count."
        )

    if args.launch_instances and (not args.love_path or not args.lovely_path):
        print(
            "WARNING: --launch-instances is enabled but --love-path/--lovely-path is missing. "
            "benchmark_balatrobot.py may fail at runtime."
        )

    args.benchmark = str(benchmark_path)


def main():
    args = parse_args()
    try:
        instances = parse_instances(args.instances)
        modes = resolve_modes(args.mode)
        validate_inputs(args, instances, modes)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 2

    raw_dir = Path(args.raw_dir).expanduser() if args.raw_dir else None
    total_runs = len(modes) * len(instances) * args.repeats
    run_counter = 0
    detail_rows = []

    for mode in modes:
        for inst in instances:
            for repeat_idx in range(1, args.repeats + 1):
                base_port_used, ports_csv = build_ports(
                    base_port=args.base_port,
                    instances=inst,
                    run_index=run_counter,
                    port_stride=args.port_stride,
                )
                cmd = build_cmd(args=args, mode=mode, instances=inst, ports_csv=ports_csv)
                cmd_text = subprocess.list2cmdline(cmd)
                print(
                    f"[{run_counter + 1}/{total_runs}] "
                    f"mode={mode} instances={inst} repeat={repeat_idx}/{args.repeats} "
                    f"base_port={base_port_used}"
                )

                run_meta = {
                    "mode": mode,
                    "instances": inst,
                    "repeat": repeat_idx,
                    "run_index": run_counter,
                    "base_port_used": base_port_used,
                }
                result = run_once(
                    cmd=cmd,
                    timeout_sec=args.timeout,
                    raw_dir=raw_dir,
                    run_meta=run_meta,
                )

                if result["timeout"]:
                    print("ERROR: benchmark run timed out.")
                    print(f"  cmd={cmd_text}")
                    if result["raw_path"]:
                        print(f"  raw_output={result['raw_path']}")
                    else:
                        print("  stdout:")
                        print(result["stdout"])
                        print("  stderr:")
                        print(result["stderr"])
                    return 1

                if result["returncode"] != 0:
                    print("ERROR: benchmark run failed (non-zero return code).")
                    print(f"  returncode={result['returncode']}")
                    print(f"  cmd={cmd_text}")
                    if result["raw_path"]:
                        print(f"  raw_output={result['raw_path']}")
                    else:
                        print("  stdout:")
                        print(result["stdout"])
                        print("  stderr:")
                        print(result["stderr"])
                    return 1

                try:
                    steps_s_total, parse_source = parse_steps_s_total(
                        stdout_text=result["stdout"],
                        instances=inst,
                    )
                except ValueError as exc:
                    print(f"ERROR: parse failure for steps/s_total: {exc}")
                    print(f"  cmd={cmd_text}")
                    if result["raw_path"]:
                        print(f"  raw_output={result['raw_path']}")
                    else:
                        print("  stdout:")
                        print(result["stdout"])
                        print("  stderr:")
                        print(result["stderr"])
                    return 1

                play_avg_ms = parse_method_avg_ms(result["stdout"], "play")
                gamestate_avg_ms = parse_method_avg_ms(result["stdout"], "gamestate")
                detail_rows.append(
                    {
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "mode": mode,
                        "instances": inst,
                        "repeat": repeat_idx,
                        "steps_s_total": steps_s_total,
                        "base_port_used": base_port_used,
                        "returncode": result["returncode"],
                        "cmd": cmd_text,
                        "parse_source": parse_source,
                        "play_avg_ms": play_avg_ms,
                        "gamestate_avg_ms": gamestate_avg_ms,
                    }
                )
                print(
                    f"  parsed steps/s_total={steps_s_total:.4f} "
                    f"(source={parse_source}, play_avg_ms={play_avg_ms}, gamestate_avg_ms={gamestate_avg_ms})"
                )
                run_counter += 1

    summary_rows = summarize_results(detail_rows)
    print_summary_table(summary_rows, modes)
    print_platform_info()

    if args.csv_out:
        csv_path = Path(args.csv_out).expanduser()
        if not csv_path.is_absolute():
            csv_path = (Path.cwd() / csv_path).resolve()
        summary_path = csv_path.with_name(f"{csv_path.stem}_summary{csv_path.suffix or '.csv'}")
        write_detail_csv(csv_path, detail_rows)
        write_summary_csv(summary_path, summary_rows)
        print(f"\nCSV written: {csv_path}")
        print(f"Summary CSV written: {summary_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())


# Example 1: direct + auto launch + action_only
# python sweep_throughput.py --mode action_only --launch-instances --launcher direct --balatrobot-cmd balatrobot --love-path "D:\SteamLibrary\steamapps\common\Balatro\Balatro.exe" --lovely-path "D:\SteamLibrary\steamapps\common\Balatro\version.dll"
#
# Example 2: direct + auto launch + both(action_only + rl_step)
# python sweep_throughput.py --mode both --launch-instances --launcher direct --balatrobot-cmd balatrobot --love-path "D:\SteamLibrary\steamapps\common\Balatro\Balatro.exe" --lovely-path "D:\SteamLibrary\steamapps\common\Balatro\version.dll"
