from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from exporters.voters_db import default_db_path
from parser.coverage import ac_parts_payload, build_coverage_tree, resolve_part_pdf
from parser.search import query_voters, voters_total

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

        def _json(self, code: int, payload: dict) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode()
            self._send(code, body, "application/json; charset=utf-8")

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path
            qs = parse_qs(parsed.query)
            if path in ("/", "/index.html"):
                self._send(200, INDEX.read_bytes(), "text/html; charset=utf-8")
                return
            if path == "/api/stats":
                total = voters_total(out_dir)
                self._json(
                    200,
                    {
                        "total": total,
                        "csv": str(out_dir / "csv" / "all_voters.csv"),
                        "db": str(default_db_path(out_dir)),
                    },
                )
                return
            if path == "/api/search":
                q = qs.get("q", [""])[0]
                district = (qs.get("district", [""])[0] or "").strip() or None
                ac = (qs.get("ac", [""])[0] or "").strip() or None
                asmbly_raw = (qs.get("asmblyNo", [""])[0] or "").strip()
                part_raw = (qs.get("part", qs.get("partNo", [""]))[0] or "").strip()
                asmbly_no = int(asmbly_raw) if asmbly_raw.isdigit() else None
                part_no = int(part_raw) if part_raw.isdigit() else None
                has_filter = bool(district or ac or asmbly_no is not None or part_no is not None)

                if not q.strip() and not has_filter:
                    count, voters = 0, []
                else:
                    limit = 500 if part_no is not None else 200
                    count, voters = query_voters(
                        out_dir,
                        q,
                        district=district,
                        ac=ac,
                        asmbly_no=asmbly_no,
                        part_no=part_no,
                        limit=limit,
                    )

                self._json(
                    200,
                    {
                        "query": q,
                        "filters": {
                            "district": district,
                            "ac": ac,
                            "asmblyNo": asmbly_no,
                            "partNo": part_no,
                        },
                        "count": count,
                        "voters": voters,
                    },
                )
                return
            if path == "/api/pdf":
                district = (qs.get("district", [""])[0] or "").strip()
                ac = (qs.get("ac", [""])[0] or "").strip()
                part_raw = (qs.get("part", [""])[0] or "").strip()
                if not district or not ac or not part_raw.isdigit():
                    self._send(400, b"district, ac, and part required", "text/plain")
                    return
                pdf = resolve_part_pdf(district, ac, int(part_raw))
                if pdf is None:
                    self._send(404, b"pdf not found", "text/plain")
                    return
                body = pdf.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "application/pdf")
                self.send_header(
                    "Content-Disposition",
                    f'inline; filename="{pdf.name}"',
                )
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if path == "/api/coverage":
                try:
                    self._json(200, build_coverage_tree(out_dir))
                except FileNotFoundError as exc:
                    self._json(500, {"error": str(exc)})
                return
            if path == "/api/coverage/ac":
                district = qs.get("district", [""])[0]
                asmbly = qs.get("asmblyNo", [""])[0]
                if not district or not asmbly:
                    self._json(400, {"error": "district and asmblyNo required"})
                    return
                try:
                    self._json(
                        200,
                        ac_parts_payload(
                            out_dir,
                            district=district,
                            asmbly_no=int(asmbly),
                        ),
                    )
                except (FileNotFoundError, ValueError) as exc:
                    self._json(500, {"error": str(exc)})
                return
            self._send(404, b"not found", "text/plain")

    return Handler


def serve(out_dir: Path, host: str = "127.0.0.1", port: int = 8765) -> None:
    httpd = ThreadingHTTPServer((host, port), make_handler(out_dir))
    print(f"Voter search: http://{host}:{port}", flush=True)
    print(f"Reading {default_db_path(out_dir)}  (Ctrl+C to stop)", flush=True)
    print(f"Coverage: {out_dir / 'coverage.json'}", flush=True)
    httpd.serve_forever()
