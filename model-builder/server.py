"""Standalone converter over HTTP(S). No auth (per requirements).

    POST /convert   body: {"arch_id", "weights":[{"shape","data"},...]}
                    -> 200 raw .tflite bytes, headers X-Convert-Ms / X-Max-Abs-Err
    GET  /          -> health check

TLS: if cert.pem + key.pem exist (or TLS_CERT/TLS_KEY point to them) it serves HTTPS;
otherwise it serves plain HTTP with a warning. Generate a self-signed pair with:

    openssl req -x509 -newkey rsa:2048 -nodes -keyout key.pem -out cert.pem \
            -days 365 -subj '/CN=localhost'

Run:  python server.py            (PORT=8443 by default)
"""
import json
import os
import ssl
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import converter

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8443"))
CERT = os.environ.get("TLS_CERT", "cert.pem")
KEY = os.environ.get("TLS_KEY", "key.pem")

CORS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Expose-Headers": "X-Convert-Ms, X-Max-Abs-Err",
}


class Handler(BaseHTTPRequestHandler):
    def _cors(self):
        for k, v in CORS.items():
            self.send_header(k, v)

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        self.send_response(200)
        self._cors()
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"ai-on-edges converter: POST /convert {arch_id, weights}\n")

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(n))
            tflite, meta = converter.convert_weights(
                payload["arch_id"], payload["weights"],
                quantize=payload.get("quantize", False), freqs=payload.get("freqs"))
            self.send_response(200)
            self._cors()
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("X-Convert-Ms", str(meta["convert_ms"]))
            self.send_header("X-Max-Abs-Err", str(meta["max_abs_err"]))
            self.send_header("Content-Length", str(len(tflite)))
            self.end_headers()
            self.wfile.write(tflite)
            print(f"[server] {meta}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            body = json.dumps({"error": str(e)}).encode()
            self.send_response(400)
            self._cors()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def log_message(self, *args):
        pass  # quiet


def main():
    httpd = ThreadingHTTPServer((HOST, PORT), Handler)
    scheme = "http"
    if os.path.exists(CERT) and os.path.exists(KEY):
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(CERT, KEY)
        httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
        scheme = "https"
    else:
        print(f"[server] no {CERT}/{KEY} -> plain HTTP (dev only). See the module docstring for TLS.")
    print(f"[server] converter on {scheme}://{HOST}:{PORT}  (POST /convert)")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
