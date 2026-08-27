

def build_upstream(body):
    ins, items = convert_messages(body.get("messages"))
    model = MODEL_MAP.get(body.get("model"), body.get("model"))
    up = {"model": model, "input": items, "store": False}
    if ins:
        up["instructions"] = ins
    tools = map_tools(body.get("tools"))
    if tools:
        up["tools"] = tools
    tc = map_tool_choice(body.get("tool_choice"))
    if tc:
        up["tool_choice"] = tc
    if body.get("temperature") is not None:
        up["temperature"] = body["temperature"]
    if body.get("top_p") is not None:
        up["top_p"] = body["top_p"]
    mt = body.get("max_tokens") or body.get("max_completion_tokens")
    if mt:
        up["max_output_tokens"] = mt
    if body.get("parallel_tool_calls") is not None:
        up["parallel_tool_calls"] = body["parallel_tool_calls"]
    if REASONING:
        up["reasoning"] = {"effort": REASONING}
    return up


def usage_from(resp_json):
    u = resp_json.get("usage") or {}
    return {
        "prompt_tokens": u.get("input_tokens", 0),
        "completion_tokens": u.get("output_tokens", 0),
        "total_tokens": u.get("total_tokens", 0),
    }


def resp_to_chat(resp_json, model, chat_id):
    texts = []
    tool_calls = []
    for o in resp_json.get("output") or []:
        t = o.get("type")
        if t == "message":
            for p in o.get("content") or []:
                if isinstance(p, dict) and p.get("type") == "output_text":
                    texts.append(p.get("text", ""))
        elif t == "function_call":
            tool_calls.append({
                "id": o.get("call_id") or o.get("id") or f"call_{len(tool_calls)}_{uuid.uuid4().hex[:6]}",
                "type": "function",
                "function": {"name": o.get("name", ""), "arguments": o.get("arguments") or "{}"},
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
