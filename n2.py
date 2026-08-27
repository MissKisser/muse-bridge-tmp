

def text_of_content(c):
    """Flatten string/array content to plain text; non-text parts are dropped."""
    if c is None:
        return ""
    if isinstance(c, str):
        return c
    parts = []
    if isinstance(c, list):
        for p in c:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict) and p.get("type") in TEXT_PART_TYPES:
                parts.append(p.get("text", ""))
            elif isinstance(p, dict) and p.get("type") == "tool_result":
                inner = text_of_content(p.get("content"))
                if inner:
                    parts.append(inner)
    if not parts and not isinstance(c, list):
        return str(c)
    return "\n".join(x for x in parts if x)


def map_user_content(c):
    if c is None:
        return [{"type": "input_text", "text": ""}]
    if isinstance(c, str):
        return [{"type": "input_text", "text": c}]
    out = []
    for p in c:
        if isinstance(p, str):
            out.append({"type": "input_text", "text": p})
            continue
        if not isinstance(p, dict):
            continue
        t = p.get("type")
        if t in TEXT_PART_TYPES:
            out.append({"type": "input_text", "text": p.get("text", "")})
        elif t == "image_url":
            u = p.get("image_url")
            u = u.get("url") if isinstance(u, dict) else u
            if u:
                out.append({"type": "input_image", "image_url": u})
        elif t == "input_image":
            u = p.get("image_url") or p.get("url")
            if u:
                out.append({"type": "input_image", "image_url": u})
        else:
            txt = text_of_content(p)
            if txt:
                out.append({"type": "input_text", "text": txt})
    return out or [{"type": "input_text", "text": ""}]


def convert_messages(messages):
    instructions = []
    items = []
    for m in messages or []:
        role = m.get("role")
        if role in ("system", "developer"):
            t = text_of_content(m.get("content"))
            if t:
                instructions.append(t)
        elif role == "user":
            items.append({"role": "user", "content": map_user_content(m.get("content"))})
        elif role == "assistant":
            txt = text_of_content(m.get("content"))
            if txt:
                items.append({"role": "assistant",
                              "content": [{"type": "output_text", "text": txt}]})
            for tc in m.get("tool_calls") or []:
                f = tc.get("function") or {}
                items.append({
                    "type": "function_call",
                    "call_id": tc.get("id") or f"call_{uuid.uuid4().hex[:8]}",
                    "name": f.get("name", ""),
                    "arguments": f.get("arguments") or "{}",
                })
        elif role == "tool":
            items.append({
                "type": "function_call_output",
                "call_id": m.get("tool_call_id", ""),
                "output": text_of_content(m.get("content")) or "",
            })
    return ("\n\n".join(instructions)), items


def safe_tool_name(name):
    n = re.sub(r"[^A-Za-z0-9_-]", "_", name or "")
    if len(n) <= 64:
        return n
    digest = hashlib.sha1((name or "").encode("utf-8")).hexdigest()[:8]
    return f"{n[:55]}_{digest}"


def map_tools(tools):
    """chat/anthropic nested form -> flat Responses form; returns (out, name_map)."""
    out = []
    name_map = {}
    for t in tools or []:
        if not isinstance(t, dict) or t.get("type") != "function":
            continue
        f = t.get("function")
        if isinstance(f, dict):
            orig = f.get("name", "")
            nt = {"type": "function", "name": safe_tool_name(orig),
                  "parameters": f.get("parameters") or {"type": "object", "properties": {}}}
            if f.get("description"):
                nt["description"] = f["description"]
            name_map[nt["name"]] = orig
            out.append(nt)
        elif t.get("name"):
            t = dict(t)
            name_map[safe_tool_name(t["name"])] = t["name"]
            t["name"] = safe_tool_name(t["name"])
            out.append(t)
    return out, name_map


def unmap_names_inplace(obj, name_map):
    """safe -> orig for every function_call name found anywhere in obj."""
    if not name_map or obj is None:
        return obj
    stack = [obj]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            if cur.get("type") == "function_call" and cur.get("name") in name_map:
                cur["name"] = name_map[cur["name"]]
            stack.extend(cur.values())
        elif isinstance(cur, list):
            stack.extend(cur)
    return obj
