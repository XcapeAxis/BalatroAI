from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import json
import webbrowser
from http.server import ThreadingHTTPServer

from demo.api import DemoApplication, create_handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="启动 BalatroAI MVP-S2 本地 Web Demo。")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--open-browser", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    host = str(args.host or "127.0.0.1")
    port = max(1, int(args.port))
    app = DemoApplication()
    server = ThreadingHTTPServer((host, port), create_handler(app))
    url = f"http://{host}:{port}/"
    print(json.dumps({"status": "ok", "url": url, "model_loaded": app.session.bundle.loaded}, ensure_ascii=False))
    if args.open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
