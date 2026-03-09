from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PATH_CATEGORY_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("readme", ("readme.md",)),
    ("agents", ("agents.md",)),
    ("docs", ("docs/",)),
    ("scripts", ("scripts/",)),
    ("configs", ("configs/",)),
    ("trainer_hybrid", ("trainer/hybrid/",)),
    ("trainer_world_model", ("trainer/world_model/",)),
    ("trainer_rl", ("trainer/rl/",)),
    ("trainer_closed_loop", ("trainer/closed_loop/",)),
    ("trainer_experiments", ("trainer/experiments/",)),
    ("trainer_runtime", ("trainer/runtime/",)),
    ("trainer_any", ("trainer/",)),
    ("sim_core", ("sim/",)),
)

HIGH_RISK_CATEGORIES = {
    "sim_core",
    "trainer_hybrid",
    "trainer_world_model",
    "trainer_rl",
    "trainer_closed_loop",
    "trainer_experiments",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_repo_root(repo_root: str | Path | None = None) -> Path:
    if repo_root:
        return Path(repo_root).resolve()
    preferred = Path("D:/MYFILES/BalatroAI")
    if preferred.exists():
        return preferred.resolve()
    return Path(__file__).resolve().parents[2]


def _run_git(repo_root: Path, args: list[str]) -> list[str]:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(repo_root),
        text=True,
        capture_output=True,
        check=False,
    )
    if int(proc.returncode or 0) != 0:
        return []
    return [line.strip().replace("\\", "/") for line in str(proc.stdout or "").splitlines() if line.strip()]


def detect_changed_files(repo_root: Path, explicit_files: list[str] | None = None) -> list[str]:
    if explicit_files:
        return sorted({str(Path(item)).replace("\\", "/") for item in explicit_files if str(item).strip()})

    staged = _run_git(repo_root, ["diff", "--name-only", "--cached"])
    unstaged = _run_git(repo_root, ["diff", "--name-only"])
    untracked = _run_git(repo_root, ["ls-files", "--others", "--exclude-standard"])
    merged = sorted({*staged, *unstaged, *untracked})
    if merged:
        return merged

    last_commit = _run_git(repo_root, ["diff", "--name-only", "HEAD~1", "HEAD"])
    if last_commit:
        return sorted(set(last_commit))
    return []


def classify_path(path_text: str) -> list[str]:
    token = str(path_text or "").strip().replace("\\", "/").lower()
    categories: list[str] = []
    for name, prefixes in PATH_CATEGORY_RULES:
        if any(token == prefix or token.startswith(prefix) for prefix in prefixes):
            categories.append(name)
    if token.endswith(".py"):
        categories.append("python_code")
    if token.endswith(".ps1"):
        categories.append("powershell")
    if token.endswith((".yaml", ".yml", ".json")):
        categories.append("config_like")
    return sorted(set(categories))


def build_change_scope(*, repo_root: str | Path | None = None, changed_files: list[str] | None = None) -> dict[str, Any]:
    root = resolve_repo_root(repo_root)
    files = detect_changed_files(root, changed_files)
    file_rows: list[dict[str, Any]] = []
    category_counts: dict[str, int] = {}
    changed_python_files: list[str] = []
    for path_text in files:
        categories = classify_path(path_text)
        for category in categories:
            category_counts[category] = int(category_counts.get(category, 0)) + 1
        if "python_code" in categories:
            changed_python_files.append(path_text)
        file_rows.append({"path": path_text, "categories": categories})

    top_level = {category for category in category_counts if category in {name for name, _ in PATH_CATEGORY_RULES}}
    docs_only = bool(files) and all(category in {"docs", "readme", "agents"} for category in top_level)
    high_risk = sorted(category for category in top_level if category in HIGH_RISK_CATEGORIES)
    return {
        "schema": "p61_change_scope_v1",
        "generated_at": _now_iso(),
        "repo_root": str(root),
        "changed_files": files,
        "file_rows": file_rows,
        "category_counts": category_counts,
        "top_level_categories": sorted(top_level),
        "changed_python_files": changed_python_files,
        "docs_only": docs_only,
        "high_risk_categories": high_risk,
        "requires_runtime_validation": any(category not in {"docs", "readme", "agents"} for category in top_level),
        "contains_p22_entrypoint_changes": any(
            path in {"scripts/run_p22.ps1", "scripts/run_regressions.ps1", "scripts/run_autonomy.ps1"} for path in files
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="P61 change-scope classifier")
    parser.add_argument("--repo-root", default="")
    parser.add_argument("--changed-file", action="append", default=[])
    args = parser.parse_args()
    payload = build_change_scope(repo_root=args.repo_root or None, changed_files=list(args.changed_file or []))
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
