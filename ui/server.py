from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from parser.search import load_voters, rows_as_dicts, search_voters

UI_DIR = Path(__file__).resolve().parent
INDEX = UI_DIR / "index.html"


def make_handler(out_dir: Path) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:
            return

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path
            if path in ("/", "/index.html"):
                self._send(200, INDEX.read_bytes(), "text/html; charset=utf-8")
                return
            if path == "/api/stats":
                df = load_voters(out_dir)
                payload = json.dumps({"total": df.height, "csv": str(out_dir / "csv" / "all_voters.csv")})
                self._send(200, payload.encode(), "application/json")
                return
            if path == "/api/search":
                q = parse_qs(parsed.query).get("q", [""])[0]
                df = load_voters(out_dir)
                hits = search_voters(df, q) if q.strip() else df.head(0)
                payload = json.dumps({"query": q, "count": hits.height, "voters": rows_as_dicts(hits.head(200))})
                self._send(200, payload.encode(), "application/json")
                return
            self._send(404, b"not found", "text/plain")

    return Handler


def serve(out_dir: Path, host: str = "127.0.0.1", port: int = 8765) -> None:
    httpd = ThreadingHTTPServer((host, port), make_handler(out_dir))
    print(f"Voter search: http://{host}:{port}", flush=True)
    print(f"Reading {out_dir / 'csv' / 'all_voters.csv'}  (Ctrl+C to stop)", flush=True)
    httpd.serve_forever()
