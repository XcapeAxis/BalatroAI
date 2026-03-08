from __future__ import annotations

import argparse
import json
import sys


def _base_probe() -> dict[str, object]:
    data: dict[str, object] = {
        "python": sys.executable,
        "python_version": sys.version.replace("\n", " ").strip(),
        "prefix": sys.prefix,
        "base_prefix": getattr(sys, "base_prefix", sys.prefix),
        "yaml_available": False,
        "yaml_version": None,
        "pip_version": None,
    }
    try:
        import yaml  # type: ignore

        data["yaml_available"] = True
        data["yaml_version"] = str(getattr(yaml, "__version__", ""))
    except Exception as exc:
        data["yaml_error"] = repr(exc)
    try:
        import pip  # type: ignore

        data["pip_version"] = str(getattr(pip, "__version__", ""))
    except Exception as exc:
        data["pip_error"] = repr(exc)
    return data


def _torch_probe() -> dict[str, object]:
    import torch  # type: ignore

    cuda = bool(torch.cuda.is_available())
    return {
        "torch_available": True,
        "torch_version": str(torch.__version__),
        "cuda_available": cuda,
        "device_count": int(torch.cuda.device_count()) if cuda else 0,
        "device_name": str(torch.cuda.get_device_name(0)) if cuda and int(torch.cuda.device_count()) > 0 else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Emit a small JSON probe for a Python environment.")
    parser.add_argument("--kind", choices=("base", "torch"), required=True)
    args = parser.parse_args()
    payload = _base_probe() if args.kind == "base" else _torch_probe()
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
