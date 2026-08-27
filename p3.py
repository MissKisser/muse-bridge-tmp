

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # silence default stderr chatter
        pass

    def _authed(self):
        if not BRIDGE_TOKEN:
            return True
        auth = self.headers.get("Authorization", "")
        return auth == f"Bearer {BRIDGE_TOKEN}"

    def _send(self, status, payload, ctype="application/json"):
        body = payload if isinstance(payload, bytes) else json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except Exception:
            pass

    def do_GET(self):
        path = self.path.rstrip("/")
        if path in ("/healthz", "/health"):
            return self._send(200, {"ok": True})
        if path.endswith("/models"):
            return self._send(200, {"object": "list",
                                    "data": [{"id": m, "object": "model", "owned_by": "muse-bridge"} for m in MODELS]})
        self._send(404, {"error": {"message": "not found"}})

    def do_POST(self):
        if "/chat/completions" not in self.path:
            return self._send(404, {"error": {"message": "not found"}})
        if not self._authed():
            return self._send(401, {"error": {"message": "unauthorized"}})
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
        except Exception as e:
            return self._send(400, {"error": {"message": f"bad request body: {e}"}})
        model = body.get("model", "")
        stream = bool(body.get("stream"))
        t0 = time.time()
        chat_id = "chatcmpl-" + uuid.uuid4().hex[:24]
        upstream_body = build_upstream(body)
        headers = {
            "Authorization": f"Bearer {OP_KEY}" if not OP_KEY.startswith("Bearer ") else OP_KEY,
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if stream else "application/json",
        }
        try:
            if not stream:
                with httpx.Client(timeout=TIMEOUT) as cli:
                    r = cli.post(UPSTREAM_URL, json=upstream_body, headers=headers)
                    if r.status_code != 200:
                        st, payload = upstream_error(r)
                        log(f"NOSTREAM {model} -> {r.status_code} ({time.time()-t0:.1f}s)")
                        return self._send(st if st >= 400 else 502, payload)
                    comp = resp_to_chat(r.json(), model, chat_id)
                    log(f"NOSTREAM {model} 200 finish={comp['choices'][0]['finish_reason']} ({time.time()-t0:.1f}s)")
                    return self._send(200, comp)
            self._do_stream(upstream_body, headers, model, chat_id, t0)
        except BrokenPipeError:
            log(f"{model} client disconnected")
        except Exception as e:
            log(f"ERROR {model}: {e!r}")
            try:
                self._send(502, {"error": {"message": f"muse-bridge internal error: {e}", "type": "bridge_error"}})
            except Exception:
                pass
