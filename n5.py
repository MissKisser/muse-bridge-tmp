

    # ---- chat-wire streaming ---------------------------------------------

    def _chunk(self, chat_id, model, delta=None, finish=None, usage=None):
        obj = {"id": chat_id, "object": "chat.completion.chunk",
               "created": int(time.time()), "model": model,
               "choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}]}
        if usage is not None:
            obj["usage"] = usage
        return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

    def do_stream(self, upstream_body, headers, model, chat_id, t0, name_map=None):
        upstream_body["stream"] = True
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        final_usage = None
        finish_reason = "stop"
        pending_tools = []

        def raw(b):
            self.wfile.write(b)
            self.wfile.flush()

        def chunk_hex(x):
            b = x.encode() if isinstance(x, str) else x
            return hex(len(b))[2:].encode() + b"\r\n" + b + b"\r\n"

        emit = lambda s: raw(chunk_hex(s))

        def flush_tool_chunk():
            nonlocal pending_tools, finish_reason
            for i, tc in enumerate(pending_tools):
                # official framing: identity frame then args frame
                emit(self._chunk(chat_id, model, delta={"tool_calls": [{
                    "index": i, "id": tc["id"], "type": "function",
                    "function": {"name": tc["function"]["name"], "arguments": ""}}]}))
                emit(self._chunk(chat_id, model, delta={"tool_calls": [{
                    "index": i, "function": {"arguments": tc["function"]["arguments"]}}]}))
            if pending_tools:
                finish_reason = "tool_calls"
            pending_tools = []

        emit(self._chunk(chat_id, model, delta={"role": "assistant"}))
        cur_ev, cur_data = None, []
        with httpx.stream("POST", UPSTREAM_URL, json=upstream_body, headers=headers, timeout=TIMEOUT) as rr:
            if rr.status_code != 200:
                rr.read()
                st, payload = upstream_error(rr)
                errmsg = payload["error"]["message"]
                emit(self._chunk(chat_id, model, delta={"content": f"[muse-bridge] upstream {st}: {errmsg}"}))
                emit(self._chunk(chat_id, model, finish="stop"))
                emit("data: [DONE]\n\n")
                raw(b"0\r\n\r\n")
                log(f"STREAM {model} UPSTREAM_ERR {st} ({time.time()-t0:.1f}s)")
                return
            for line in rr.iter_lines():
                line = line.decode("utf-8", "replace") if isinstance(line, bytes) else line
                if line.startswith("event:"):
                    cur_ev = line[6:].strip()
                    continue
                if line.startswith("data:"):
                    cur_data.append(line[5:].strip())
                    continue
                if not line.strip():
                    ev, data = cur_ev, "".join(cur_data)
                    cur_ev, cur_data = None, []
                    if not data:
                        continue
                    try:
                        d = json.loads(data)
                    except Exception:
                        continue
                    typ = d.get("type") or ev
                    if typ == "response.output_text.delta":
                        delta = d.get("delta")
                        if delta:
                            flush_tool_chunk()
                            emit(self._chunk(chat_id, model, delta={"content": delta}))
                    elif typ in ("response.output_item.done", "response.output_item.completed"):
                        item = d.get("item") or {}
                        if item.get("type") == "function_call":
                            disp = (name_map or {}).get(item.get("name", ""), item.get("name", ""))
                            pending_tools.append({"id": item.get("call_id") or item.get("id") or f"call_{len(pending_tools)}",
                                                  "type": "function",
                                                  "function": {"name": disp,
                                                               "arguments": item.get("arguments") or "{}"}})
                    elif typ in ("response.completed", "response.done"):
                        final_usage = usage_from(d.get("response") or {})
                    elif typ in ("error", "response.failed"):
                        e = d.get("error") or d
                        msgv = e.get("message") if isinstance(e, dict) else str(e)
                        if msgv:
                            flush_tool_chunk()
                            emit(self._chunk(chat_id, model, delta={"content": f"\n[muse-bridge] upstream error: {msgv}"}))
        flush_tool_chunk()
        emit(self._chunk(chat_id, model, finish=finish_reason, usage=final_usage or {}))
        emit("data: [DONE]\n\n")
        raw(b"0\r\n\r\n")
        log(f"STREAM {model} 200 finish={finish_reason} usage={final_usage} ({time.time()-t0:.1f}s)")

    # ---- native Responses passthrough -------------------------------------

    def handle_responses_post(self):
        if not self._authed():
            return self._send(401, {"error": {"message": "unauthorized"}})
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
        except Exception as e:
            return self._send(400, {"error": {"message": f"bad request body: {e}"}})
        stream = bool(body.get("stream"))
        t0 = time.time()
        model = body.get("model", "")
        log("NATIVE-IN", model, "stream=", stream)
        body, name_map = sanitize_responses_body(body)
        headers = {
            "Authorization": f"Bearer {OP_KEY}" if not OP_KEY.startswith("Bearer ") else OP_KEY,
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if stream else "application/json",
            "User-Agent": "curl/7.81.0",
            "Accept-Encoding": "identity",
        }
        try:
            if not stream:
                with httpx.Client(timeout=TIMEOUT) as cli:
                    r = cli.post(UPSTREAM_URL, json=body, headers=headers)
                    log(f"NATIVE {model} -> {r.status_code} ({time.time()-t0:.1f}s)")
                    if r.status_code == 200:
                        data = r.json()
                        unmap_names_inplace(data, name_map)
                        return self._send(200, data)
                    st, payload = upstream_error(r)
                    return self._send(st if st >= 400 else 502, payload)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

            def raw(b):
                self.wfile.write(b)
                self.wfile.flush()

            def emit(x):
                b = x.encode() if isinstance(x, str) else x
                raw(hex(len(b))[2:].encode() + b"\r\n" + b + b"\r\n")

            n_lines = 0
            buf = ""
            with httpx.stream("POST", UPSTREAM_URL, json=body, headers=headers, timeout=TIMEOUT) as rr:
                log("UPSTATUS", rr.status_code)
                if rr.status_code != 200:
                    rr.read()
                    st, payload = upstream_error(rr)
                    emit(json.dumps(payload, ensure_ascii=False))
                    raw(b"0\r\n\r\n")
                    log(f"NATIVE {model} UPSTREAM_ERR {st}")
                    return
                # byte-level passthrough preserving upstream SSE structure;
                # safe tool names rewritten back to originals inline
                for chunk in rr.iter_text():
                    if not chunk:
                        continue
                    for k, v in (name_map or {}).items():
                        if k in chunk:
                            chunk = chunk.replace(k, v)
                    buf += chunk
                    while "\n" in buf:
                        out, buf = buf.split("\n", 1)
                        n_lines += 1
                        emit(out + "\n")
                if buf:
                    emit(buf)
            raw(b"0\r\n\r\n")
            log(f"NATIVE-STREAM {model} 200 lines={n_lines} ({time.time()-t0:.1f}s)")
        except BrokenPipeError:
            log(f"NATIVE {model} client gone")
        except Exception as e:
            log(f"NATIVE ERROR {model}: {e!r}")


if __name__ == "__main__":
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    srv.daemon_threads = True
    log(f"muse-bridge listening :{PORT} -> {UPSTREAM_URL} models={MODELS}")
    srv.serve_forever()
