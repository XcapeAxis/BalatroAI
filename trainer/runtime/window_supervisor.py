from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import ctypes
import ctypes.wintypes
import csv
import io
import json
import locale
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

try:  # pragma: no cover - optional dependency in local env
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _now_token() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None


def _read_yaml_or_json(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is not None:
            payload = yaml.safe_load(text)
        else:
            sidecar = path.with_suffix(".json")
            if not sidecar.exists():
                raise RuntimeError(f"PyYAML unavailable and no sidecar JSON for: {path}")
            payload = json.loads(sidecar.read_text(encoding="utf-8-sig"))
    else:
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(f"expected mapping config: {path}")
    return payload


def _runtime_defaults() -> dict[str, Any]:
    path = _repo_root() / "configs" / "runtime" / "runtime_defaults.yaml"
    payload = _read_yaml_or_json(path)
    defaults = payload.get("defaults")
    return defaults if isinstance(defaults, dict) else {}


def _artifacts_root() -> Path:
    return _repo_root() / "docs" / "artifacts" / "p53" / "window_supervisor"


def _latest_root() -> Path:
    return _artifacts_root() / "latest"


def _restore_state_path() -> Path:
    return _latest_root() / "restore_state.json"


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    token = str(value).strip().lower()
    if token in {"1", "true", "yes", "y", "on"}:
        return True
    if token in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _process_table() -> dict[int, dict[str, str]]:
    if os.name != "nt":
        return {}
    try:
        proc = subprocess.run(
            ["tasklist", "/fo", "csv", "/nh"],
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
            encoding=locale.getpreferredencoding(False) or "mbcs",
            errors="replace",
        )
    except Exception:
        return {}
    raw = str(proc.stdout or "")
    if not raw.strip():
        return {}
    reader = csv.reader(io.StringIO(raw))
    mapping: dict[int, dict[str, str]] = {}
    for row in reader:
        if len(row) < 2:
            continue
        try:
            pid = int(row[1])
        except Exception:
            continue
        image = row[0]
        process_name = image.rsplit(".", 1)[0]
        mapping[pid] = {
            "image_name": image,
            "process_name": process_name,
        }
    return mapping


if os.name == "nt":
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    wintypes = ctypes.wintypes

    class POINT(ctypes.Structure):
        _fields_ = [
            ("x", wintypes.LONG),
            ("y", wintypes.LONG),
        ]

    class RECT(ctypes.Structure):
        _fields_ = [
            ("left", wintypes.LONG),
            ("top", wintypes.LONG),
            ("right", wintypes.LONG),
            ("bottom", wintypes.LONG),
        ]

    class WINDOWPLACEMENT(ctypes.Structure):
        _fields_ = [
            ("length", wintypes.UINT),
            ("flags", wintypes.UINT),
            ("showCmd", wintypes.UINT),
            ("ptMinPosition", POINT),
            ("ptMaxPosition", POINT),
            ("rcNormalPosition", RECT),
        ]

    EnumWindowsProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    EnumWindows = user32.EnumWindows
    EnumWindows.argtypes = [EnumWindowsProc, wintypes.LPARAM]
    EnumWindows.restype = wintypes.BOOL

    GetWindowTextLengthW = user32.GetWindowTextLengthW
    GetWindowTextLengthW.argtypes = [wintypes.HWND]
    GetWindowTextLengthW.restype = ctypes.c_int

    GetWindowTextW = user32.GetWindowTextW
    GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    GetWindowTextW.restype = ctypes.c_int

    GetClassNameW = user32.GetClassNameW
    GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    GetClassNameW.restype = ctypes.c_int

    IsWindowVisible = user32.IsWindowVisible
    IsWindowVisible.argtypes = [wintypes.HWND]
    IsWindowVisible.restype = wintypes.BOOL

    GetWindowThreadProcessId = user32.GetWindowThreadProcessId
    GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    GetWindowThreadProcessId.restype = wintypes.DWORD

    GetWindowRect = user32.GetWindowRect
    GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(RECT)]
    GetWindowRect.restype = wintypes.BOOL

    GetWindowPlacement = user32.GetWindowPlacement
    GetWindowPlacement.argtypes = [wintypes.HWND, ctypes.POINTER(WINDOWPLACEMENT)]
    GetWindowPlacement.restype = wintypes.BOOL

    MoveWindow = user32.MoveWindow
    MoveWindow.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, wintypes.BOOL]
    MoveWindow.restype = wintypes.BOOL

    ShowWindow = user32.ShowWindow
    ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
    ShowWindow.restype = wintypes.BOOL

    SetWindowPos = user32.SetWindowPos
    SetWindowPos.argtypes = [
        wintypes.HWND,
        wintypes.HWND,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_uint,
    ]
    SetWindowPos.restype = wintypes.BOOL

    SW_HIDE = 0
    SW_SHOWNORMAL = 1
    SW_SHOWMINIMIZED = 2
    SW_SHOW = 5
    SW_MINIMIZE = 6
    SW_SHOWMINNOACTIVE = 7
    SW_RESTORE = 9
    SW_SHOWDEFAULT = 10
    SW_FORCEMINIMIZE = 11

    HWND_TOP = wintypes.HWND(0)
    HWND_BOTTOM = wintypes.HWND(1)
    SWP_NOSIZE = 0x0001
    SWP_NOMOVE = 0x0002
    SWP_NOZORDER = 0x0004
    SWP_NOACTIVATE = 0x0010
    SWP_SHOWWINDOW = 0x0040

    SM_XVIRTUALSCREEN = 76
    SM_YVIRTUALSCREEN = 77
    SM_CXVIRTUALSCREEN = 78
    SM_CYVIRTUALSCREEN = 79


WINDOW_MODES = ("visible", "minimized", "hidden", "offscreen", "restore")
MINIMIZED_SHOW_CMDS = {SW_SHOWMINIMIZED, SW_MINIMIZE, SW_SHOWMINNOACTIVE, SW_FORCEMINIMIZE} if os.name == "nt" else set()


@dataclass
class WindowRect:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return max(0, int(self.right) - int(self.left))

    @property
    def height(self) -> int:
        return max(0, int(self.bottom) - int(self.top))


@dataclass
class WindowInfo:
    hwnd: int
    hwnd_hex: str
    pid: int
    process_name: str
    image_name: str
    title: str
    class_name: str
    visible: bool
    show_cmd: int
    rect: WindowRect
    role: str
    manageable: bool
    mode: str

    def key(self) -> str:
        return f"{self.pid}:{self.hwnd_hex}:{self.role}"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["rect"] = asdict(self.rect)
        return payload


@dataclass
class WindowSelector:
    pid: int = 0
    process_names: tuple[str, ...] = ()
    title_contains: str = ""
    include_auxiliary: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "pid": int(self.pid),
            "process_names": list(self.process_names),
            "title_contains": self.title_contains,
            "include_auxiliary": bool(self.include_auxiliary),
        }


def _virtual_screen() -> dict[str, int]:
    if os.name != "nt":
        return {"left": 0, "top": 0, "width": 0, "height": 0, "right": 0, "bottom": 0}
    left = int(user32.GetSystemMetrics(SM_XVIRTUALSCREEN))
    top = int(user32.GetSystemMetrics(SM_YVIRTUALSCREEN))
    width = int(user32.GetSystemMetrics(SM_CXVIRTUALSCREEN))
    height = int(user32.GetSystemMetrics(SM_CYVIRTUALSCREEN))
    return {
        "left": left,
        "top": top,
        "width": width,
        "height": height,
        "right": left + width,
        "bottom": top + height,
    }


def _window_text(hwnd: int) -> str:
    if os.name != "nt":
        return ""
    length = int(GetWindowTextLengthW(hwnd))
    buf = ctypes.create_unicode_buffer(max(256, length + 1))
    GetWindowTextW(hwnd, buf, len(buf))
    return buf.value


def _class_name(hwnd: int) -> str:
    if os.name != "nt":
        return ""
    buf = ctypes.create_unicode_buffer(256)
    GetClassNameW(hwnd, buf, len(buf))
    return buf.value


def _placement(hwnd: int) -> int:
    if os.name != "nt":
        return 0
    placement = WINDOWPLACEMENT()
    placement.length = ctypes.sizeof(WINDOWPLACEMENT)
    if not GetWindowPlacement(hwnd, ctypes.byref(placement)):
        return 0
    return int(placement.showCmd)


def _window_rect(hwnd: int) -> WindowRect:
    if os.name != "nt":
        return WindowRect(0, 0, 0, 0)
    rect = RECT()
    if not GetWindowRect(hwnd, ctypes.byref(rect)):
        return WindowRect(0, 0, 0, 0)
    return WindowRect(int(rect.left), int(rect.top), int(rect.right), int(rect.bottom))


def _classify_window(process_name: str, title: str, class_name: str, visible: bool) -> tuple[str, bool]:
    pname = str(process_name or "").strip().lower()
    t = str(title or "").strip().lower()
    cls = str(class_name or "").strip()
    if pname == "balatro" and cls == "SDL_app" and t == "balatro":
        return "game_main", True
    if pname == "balatro" and cls == "ConsoleWindowClass" and t.startswith("lovely"):
        return "diagnostic_console", False
    if pname == "uvx" and cls == "ConsoleWindowClass":
        return "launcher_console", False
    if cls in {"NVOpenGLPbuffer", "MSCTFIME UI", "IME"}:
        return "auxiliary", False
    if pname in {"balatro", "love"} and visible:
        return "other_balatro", True
    if "balatro" in t or "lÖve" in t or "love" in t:
        return "title_match", True
    return "unmanaged", False


def _window_mode(visible: bool, show_cmd: int, rect: WindowRect) -> str:
    if not visible:
        return "hidden"
    if int(show_cmd) in MINIMIZED_SHOW_CMDS:
        return "minimized"
    virtual = _virtual_screen()
    if rect.right < virtual["left"] or rect.left > virtual["right"] or rect.bottom < virtual["top"] or rect.top > virtual["bottom"]:
        return "offscreen"
    return "visible"


def _enumerate_windows() -> list[WindowInfo]:
    if os.name != "nt":
        return []
    proc_map = _process_table()
    items: list[WindowInfo] = []

    def _callback(hwnd: int, lparam: int) -> bool:
        pid_value = wintypes.DWORD()
        GetWindowThreadProcessId(hwnd, ctypes.byref(pid_value))
        pid = int(pid_value.value or 0)
        if pid <= 0:
            return True
        proc = proc_map.get(pid) or {}
        title = _window_text(hwnd)
        class_name = _class_name(hwnd)
        process_name = str(proc.get("process_name") or "")
        image_name = str(proc.get("image_name") or "")
        visible = bool(IsWindowVisible(hwnd))
        role, manageable = _classify_window(process_name, title, class_name, visible)
        rect = _window_rect(hwnd)
        show_cmd = _placement(hwnd)
        items.append(
            WindowInfo(
                hwnd=int(hwnd),
                hwnd_hex=f"0x{int(hwnd):X}",
                pid=pid,
                process_name=process_name,
                image_name=image_name,
                title=title,
                class_name=class_name,
                visible=visible,
                show_cmd=show_cmd,
                rect=rect,
                role=role,
                manageable=manageable,
                mode=_window_mode(visible, show_cmd, rect),
            )
        )
        return True

    EnumWindows(EnumWindowsProc(_callback), 0)
    items.sort(key=lambda item: (item.pid, item.hwnd))
    return items


def list_windows(selector: WindowSelector | None = None) -> list[WindowInfo]:
    selector = selector or WindowSelector(process_names=("Balatro",))
    wanted = {token.strip().lower() for token in selector.process_names if str(token).strip()}
    title_token = str(selector.title_contains or "").strip().lower()
    matched: list[WindowInfo] = []
    for item in _enumerate_windows():
        if int(selector.pid) > 0 and item.pid != int(selector.pid):
            continue
        if wanted and item.process_name.strip().lower() not in wanted:
            continue
        if title_token and title_token not in item.title.lower():
            continue
        if not selector.include_auxiliary and not item.manageable:
            continue
        matched.append(item)
    return matched


def _load_restore_state() -> dict[str, Any]:
    payload = _read_json(_restore_state_path())
    return payload if isinstance(payload, dict) else {"windows": {}}


def _save_restore_state(payload: dict[str, Any]) -> None:
    payload = dict(payload)
    payload["updated_at"] = _now_iso()
    _write_json(_restore_state_path(), payload)


def _capture_restore_state(windows: list[WindowInfo]) -> None:
    payload = _load_restore_state()
    entries = payload.get("windows")
    if not isinstance(entries, dict):
        entries = {}
    for item in windows:
        if item.mode == "offscreen":
            # Keep the last on-screen position so restore returns to a meaningful rect.
            continue
        entries[item.key()] = {
            "hwnd_hex": item.hwnd_hex,
            "pid": item.pid,
            "title": item.title,
            "class_name": item.class_name,
            "role": item.role,
            "mode": item.mode,
            "visible": item.visible,
            "show_cmd": item.show_cmd,
            "rect": asdict(item.rect),
            "captured_at": _now_iso(),
        }
    payload["windows"] = entries
    _save_restore_state(payload)


def _drop_restore_state(keys: list[str]) -> None:
    payload = _load_restore_state()
    entries = payload.get("windows")
    if not isinstance(entries, dict):
        return
    changed = False
    for key in keys:
        if key in entries:
            changed = True
            entries.pop(key, None)
    if changed:
        payload["windows"] = entries
        _save_restore_state(payload)


def _restore_entry(info: WindowInfo) -> dict[str, Any] | None:
    payload = _load_restore_state()
    entries = payload.get("windows")
    if not isinstance(entries, dict):
        return None
    direct = entries.get(info.key())
    if isinstance(direct, dict):
        return direct
    for value in entries.values():
        if not isinstance(value, dict):
            continue
        if str(value.get("title") or "") == info.title and str(value.get("class_name") or "") == info.class_name:
            return value
    return None


def _set_window_pos(hwnd: int, x: int, y: int, width: int, height: int, *, bottom: bool = False) -> None:
    moved = bool(MoveWindow(hwnd, int(x), int(y), int(width), int(height), True))
    if not moved:
        insert_after = HWND_BOTTOM if bottom else HWND_TOP
        ok = bool(SetWindowPos(hwnd, insert_after, int(x), int(y), int(width), int(height), SWP_NOACTIVATE | SWP_SHOWWINDOW))
        if not ok:
            raise OSError(f"MoveWindow/SetWindowPos failed for hwnd={hwnd}")


def _apply_offscreen(info: WindowInfo) -> None:
    rect = info.rect
    width = max(rect.width, 640)
    height = max(rect.height, 480)
    virtual = _virtual_screen()
    target_x = int(virtual["left"]) - width - 64
    target_y = int(virtual["top"]) + 64
    ShowWindow(info.hwnd, SW_RESTORE)
    _set_window_pos(info.hwnd, target_x, target_y, width, height, bottom=True)


def _apply_visible(info: WindowInfo) -> None:
    restore = _restore_entry(info)
    ShowWindow(info.hwnd, SW_RESTORE)
    if isinstance(restore, dict):
        rect = restore.get("rect") if isinstance(restore.get("rect"), dict) else {}
        left = int(rect.get("left") or 120)
        top = int(rect.get("top") or 120)
        right = int(rect.get("right") or (left + 1280))
        bottom = int(rect.get("bottom") or (top + 720))
        _set_window_pos(info.hwnd, left, top, max(640, right - left), max(480, bottom - top), bottom=False)
        _drop_restore_state([info.key()])
        return
    virtual = _virtual_screen()
    width = max(info.rect.width, 1280)
    height = max(info.rect.height, 720)
    left = int(virtual["left"]) + 80
    top = int(virtual["top"]) + 80
    _set_window_pos(info.hwnd, left, top, width, height, bottom=False)


def _apply_mode(info: WindowInfo, mode: str) -> None:
    token = str(mode or "").strip().lower()
    if token == "visible":
        _apply_visible(info)
        return
    if token == "restore":
        _apply_visible(info)
        return
    if token == "minimized":
        ShowWindow(info.hwnd, SW_MINIMIZE)
        return
    if token == "hidden":
        ShowWindow(info.hwnd, SW_HIDE)
        return
    if token == "offscreen":
        _apply_offscreen(info)
        return
    raise ValueError(f"unsupported window mode: {mode}")


def _state_payload(selector: WindowSelector, windows: list[WindowInfo], *, requested_mode: str = "", operation_success: bool = True, error_reason: str = "") -> dict[str, Any]:
    return {
        "schema": "p53_window_supervisor_v1",
        "generated_at": _now_iso(),
        "target_window": selector.to_dict(),
        "requested_mode": requested_mode,
        "operation_success": bool(operation_success),
        "error_reason": str(error_reason or ""),
        "virtual_screen": _virtual_screen(),
        "matched_count": len(windows),
        "windows": [item.to_dict() for item in windows],
    }


def _write_operation_artifacts(run_dir: Path, payload: dict[str, Any]) -> None:
    state_path = run_dir / "window_state.json"
    log_path = run_dir / "window_ops.log"
    latest_state_path = _latest_root() / "window_state.json"
    latest_log_path = _latest_root() / "window_ops.log"
    _write_json(state_path, payload)
    _write_json(latest_state_path, payload)
    _append_jsonl(log_path, payload)
    _append_jsonl(latest_log_path, payload)


def inspect_windows(*, selector: WindowSelector | None = None, out_root: Path | None = None) -> dict[str, Any]:
    selector = selector or WindowSelector(process_names=("Balatro",))
    windows = list_windows(selector)
    payload = _state_payload(selector, windows)
    out_root = out_root or _artifacts_root()
    run_dir = out_root / _now_token()
    _write_operation_artifacts(run_dir, payload)
    payload["artifact_dir"] = str(run_dir.resolve())
    payload["state_path"] = str((run_dir / "window_state.json").resolve())
    payload["log_path"] = str((run_dir / "window_ops.log").resolve())
    return payload


def set_window_mode(
    mode: str,
    *,
    selector: WindowSelector | None = None,
    out_root: Path | None = None,
) -> dict[str, Any]:
    if os.name != "nt":
        raise RuntimeError("window supervisor is currently implemented for Windows only")
    token = str(mode or "").strip().lower()
    if token not in WINDOW_MODES:
        raise ValueError(f"unsupported window mode: {mode}")
    selector = selector or WindowSelector(process_names=("Balatro",))
    before = list_windows(selector)
    errors: list[str] = []
    if before and token in {"minimized", "hidden", "offscreen"}:
        _capture_restore_state(before)
    if not before:
        errors.append("no_windows_matched")
    for item in before:
        try:
            _apply_mode(item, token)
        except Exception as exc:  # pragma: no cover - Win32 behavior varies by host
            errors.append(f"{item.hwnd_hex}:{exc!r}")
    if before:
        time.sleep(0.35)
    after = list_windows(selector)
    payload = {
        "schema": "p53_window_supervisor_v1",
        "generated_at": _now_iso(),
        "requested_mode": token,
        "target_window": selector.to_dict(),
        "operation_success": len(errors) == 0 and len(before) > 0,
        "error_reason": "; ".join(errors),
        "matched_count": len(before),
        "window_mode_before": [item.to_dict() for item in before],
        "window_mode_after": [item.to_dict() for item in after],
        "virtual_screen": _virtual_screen(),
    }
    out_root = out_root or _artifacts_root()
    run_dir = out_root / _now_token()
    _write_operation_artifacts(run_dir, payload)
    payload["artifact_dir"] = str(run_dir.resolve())
    payload["state_path"] = str((run_dir / "window_state.json").resolve())
    payload["log_path"] = str((run_dir / "window_ops.log").resolve())
    return payload


def latest_window_state_path() -> Path:
    return _latest_root() / "window_state.json"


def latest_restore_state_path() -> Path:
    return _restore_state_path()


def default_window_settings() -> dict[str, Any]:
    defaults = _runtime_defaults()
    return {
        "window_mode": str(defaults.get("window_mode") or "visible"),
        "window_mode_fallback": str(defaults.get("window_mode_fallback") or "offscreen"),
        "window_restore_on_failure": _safe_bool(defaults.get("window_restore_on_failure"), True),
        "window_restore_on_exit": _safe_bool(defaults.get("window_restore_on_exit"), True),
        "validate_background_mode_before_run": _safe_bool(defaults.get("validate_background_mode_before_run"), False),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P53 Windows window supervisor")
    parser.add_argument("--mode", choices=WINDOW_MODES, default="")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--pid", type=int, default=0)
    parser.add_argument("--process-name", action="append", default=[])
    parser.add_argument("--title-contains", default="")
    parser.add_argument("--include-auxiliary", action="store_true")
    parser.add_argument("--out-root", default="")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    process_names = tuple(str(item) for item in (args.process_name or []) if str(item).strip())
    if not process_names:
        process_names = ("Balatro",)
    selector = WindowSelector(
        pid=max(0, int(args.pid or 0)),
        process_names=process_names,
        title_contains=str(args.title_contains or ""),
        include_auxiliary=bool(args.include_auxiliary),
    )
    out_root = Path(args.out_root).resolve() if str(args.out_root or "").strip() else _artifacts_root()
    if args.list or not str(args.mode or "").strip():
        payload = inspect_windows(selector=selector, out_root=out_root)
    else:
        payload = set_window_mode(str(args.mode or ""), selector=selector, out_root=out_root)
    print(json.dumps(payload, ensure_ascii=False, indent=(2 if args.json else None)))
    return 0 if bool(payload.get("operation_success", True)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
