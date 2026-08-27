

def build_upstream(body):
    ins, items = convert_messages(body.get("messages"))
    mdl = body.get("model", "")
    forced = None
    for sfx, lv in THINK_SUFFIXES:
        if mdl.endswith(sfx):
            mdl = mdl[: -len(sfx)]
            forced = lv
            break
    model = MODEL_MAP.get(mdl, mdl)
    up = {"model": model, "input": items, "store": False}
    if ins:
        up["instructions"] = ins
    tools, tool_name_map = map_tools(body.get("tools"))
    tc = body.get("tool_choice")
    if tc == "none":
        tools = []          # upstream can't suppress tools; emulate
    elif isinstance(tc, dict):
        fname = ((tc.get("function") or {}).get("name")) or tc.get("name")
        if fname and tools:
            safe = next((k for k, v in tool_name_map.items() if v == fname), None)
            tools.sort(key=lambda x: 0 if x.get("name") == safe else 1)
    if tools:
        up["tools"] = tools
    # upstream only honours tool_choice "auto"; omit every other form
    if body.get("temperature") is not None:
        up["temperature"] = body["temperature"]
    if body.get("top_p") is not None:
        up["top_p"] = body["top_p"]
    mt = body.get("max_tokens") or body.get("max_completion_tokens")
    MIN_OUT = int(os.environ.get("MIN_OUTPUT", "512"))
    if mt:
        # muse burns hidden reasoning tokens first; tiny caps yield an empty
        # message with finish=length. Floor the cap so short requests still work.
        up["max_output_tokens"] = max(int(mt), MIN_OUT)
    if body.get("parallel_tool_calls") is not None:
        up["parallel_tool_calls"] = body["parallel_tool_calls"]
    effort = resolve_effort(body, forced)
    if effort:
        up["reasoning"] = reasoning_block(effort)
        log(f"EFFORT model={body.get('model')} -> {effort}")
    return up, tool_name_map


def usage_from(resp_json):
    u = resp_json.get("usage") or {}
    return {
        "prompt_tokens": u.get("input_tokens", 0),
        "completion_tokens": u.get("output_tokens", 0),
        "total_tokens": u.get("total_tokens", 0),
    }


def resp_to_chat(resp_json, model, chat_id, name_map=None):
    texts = []
    tool_calls = []
    for o in resp_json.get("output") or []:
        t = o.get("type")
        if t == "message":
            for p in o.get("content") or []:
                if isinstance(p, dict) and p.get("type") == "output_text":
                    texts.append(p.get("text", ""))
        elif t == "function_call":
            nm = o.get("name", "")
            disp = (name_map or {}).get(nm, nm)
            tool_calls.append({
                "id": o.get("call_id") or o.get("id") or f"call_{len(tool_calls)}_{uuid.uuid4().hex[:6]}",
                "type": "function",
                "function": {"name": disp, "arguments": o.get("arguments") or "{}"},
            })
    finish = "stop"
    inc = resp_json.get("incomplete_details") or {}
    if tool_calls:
        finish = "tool_calls"
    elif inc.get("reason"):
        finish = "length"
    msg = {"role": "assistant", "content": "\n".join(texts)}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {
        "id": chat_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
        "usage": usage_from(resp_json),
    }


def upstream_error(resp):
    try:
        data = resp.json()
    except Exception:
        try:
            data = {"message": resp.text[:500]}
        except Exception:
            data = {"message": f"upstream HTTP {resp.status_code}"}
    err = data.get("error")
    msg = err.get("message") if isinstance(err, dict) and err.get("message") else data.get("message") or json.dumps(data)[:500]
    typ = err.get("type") if isinstance(err, dict) else None
    return resp.status_code, {"error": {"message": msg, "type": typ or "upstream_error", "code": None}}


def sanitize_responses_body(body):
    """Native Responses passthrough sanitizer: aliases, effort+summary,
    tool naming rails, unsupported fields removed."""
    mdl = body.get("model", "")
    forced = None
    for sfx, lv in THINK_SUFFIXES:
        if mdl.endswith(sfx):
            mdl = mdl[: -len(sfx)]
            forced = lv
            break
    body["model"] = MODEL_MAP.get(mdl, mdl)
    eff = resolve_effort(body, forced)
    if eff:
        rc = body.get("reasoning")
        if isinstance(rc, dict):
            rc["effort"] = eff
            rc.setdefault("summary", "auto")
        else:
            body["reasoning"] = reasoning_block(eff)
    tcs = body.get("tools")
    name_map = {}
    if isinstance(tcs, list):
        fixed = []
        for t in tcs:
            if isinstance(t, dict) and t.get("type") == "function":
                orig = t.get("name", "")
                safe = safe_tool_name(orig)
                nt = dict(t)
                nt["name"] = safe
                if orig != safe:
                    name_map[safe] = orig
                fixed.append(nt)
            else:
                fixed.append(t)
        body["tools"] = fixed
        tc = body.get("tool_choice")
        if isinstance(tc, dict):
            fname = ((tc.get("function") or {}).get("name")) or tc.get("name")
            if fname:
                safe = next((k for k, v in name_map.items() if v == fname), None)
                tcs.sort(key=lambda x: 0 if isinstance(x, dict) and x.get("name") == safe else 1)
    if "tool_choice" in body and body["tool_choice"] != "auto":
        body.pop("tool_choice", None)   # upstream supports only auto
    mt = body.get("max_output_tokens")
    MIN_OUT = int(os.environ.get("MIN_OUTPUT", "512"))
    if mt:
        body["max_output_tokens"] = max(int(mt), MIN_OUT)
    body["store"] = False
    return body, name_map
