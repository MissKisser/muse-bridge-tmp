

    def _chunk(self, chat_id, model, delta=None, finish=None, usage=None):
        obj = {"id": chat_id, "object": "chat.completion.chunk",
               "created": int(time.time()), "model": model,
               "choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}]}
        if usage is not None:
            obj["usage"] = usage
        return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

    def _do_stream(self, upstream_body, headers, model, chat_id, t0):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        final_usage = None
        finish_reason = "stop"
        pending_tools = []   # completed function_call snapshots waiting to be flushed

        def raw(b):
            self.wfile.write(b)
            self.wfile.flush()

        def chunk_hex(s):
            b = s.encode()
            return hex(len(b))[2:].encode() + b"\r\n" + b + b"\r\n"

        emit = lambda s: raw(chunk_hex(s))

        def flush_tool_chunk():
            nonlocal pending_tools, finish_reason
            if pending_tools:
                deltas = []
                for i, tc in enumerate(pending_tools):
                    deltas.append({"index": i, "id": tc["id"], "type": "function",
                                   "function": tc["function"]})
                emit(self._chunk(chat_id, model, delta={"tool_calls": deltas}))
                pending_tools = []
                finish_reason = "tool_calls"

        cur_ev = None
        cur_data = []
        with httpx.stream("POST", UPSTREAM_URL, json=upstream_body, headers=headers, timeout=TIMEOUT) as r:
            if r.status_code != 200:
                r.read()
                st, payload = upstream_error(r)
                # mid-response failure: emit as text so agent UIs see it, then close
                errmsg = payload["error"]["message"]
                emit(self._chunk(chat_id, model, delta={"role": "assistant"}))
                emit(self._chunk(chat_id, model, delta={"content": f"[muse-bridge] upstream {st}: {errmsg}"}))
                emit(self._chunk(chat_id, model, finish="stop"))
                emit("data: [DONE]\n\n")
                raw(b"0\r\n\r\n")
                log(f"STREAM {model} UPSTREAM_ERR {st} ({time.time()-t0:.1f}s)")
                return
            for line in r.iter_lines():
                line = line.decode("utf-8", "replace") if isinstance(line, bytes) else line
                if line.startswith("event:"):
                    cur_ev = line[6:].strip()
                    continue
                if line.startswith("data:"):
                    cur_data.append(line[5:].strip())
                    continue
                if not line.strip():
                    ev = cur_ev
                    data = "".join(cur_data)
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
                            flush_tool_chunk()  # keep text-after-tools ordering simple
                            emit(self._chunk(chat_id, model, delta={"content": delta}))
                    elif typ in ("response.output_item.done", "response.output_item.completed"):
                        item = d.get("item") or {}
                        if item.get("type") == "function_call":
                            tc_id = item.get("call_id") or item.get("id") or f"call_{len(pending_tools)}"
                            pending_tools.append({"id": tc_id, "type": "function",
                                                  "function": {"name": item.get("name", ""),
                                                               "arguments": item.get("arguments") or "{}"}})
                    elif typ in ("response.completed", "response.done"):
                        rr = d.get("response") or {}
                        final_usage = usage_from(rr)
                    elif typ in ("error", "response.failed"):
                        e = d.get("error") or d
                        msgv = e.get("message") if isinstance(e, dict) else str(e)
                        if msgv:
                            emit(self._chunk(chat_id, model, delta={"content": f"\n[muse-bridge] upstream error: {msgv}"}))
        flush_tool_chunk()
        emit(self._chunk(chat_id, model, delta={}, finish=finish_reason))
        if final_usage is not None:
            emit(self._chunk(chat_id, model, usage=final_usage))
        emit("data: [DONE]\n\n")
        raw(b"0\r\n\r\n")
        log(f"STREAM {model} 200 finish={finish_reason} usage={final_usage} ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    srv.daemon_threads = True
    log(f"muse-bridge listening :{PORT} -> {UPSTREAM_URL} models={MODELS}")
    srv.serve_forever()
