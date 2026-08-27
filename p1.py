#!/usr/bin/env python3
"""muse-bridge: OpenAI Chat Completions API (inbound) -> OpenAI Responses API (upstream).

Sits between new-api and OpenCode Zen: the gateway speaks plain chat-completions
(JSON or SSE), the bridge translates to/from the Responses wire format that
muse-family models require. Upstream auth always comes from OP_KEY; inbound
Authorization is ignored unless BRIDGE_TOKEN is set.
"""
import json
import os
import sys
import time
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx

OP_BASE = os.environ.get("OP_BASE", "https://opencode.ai/zen/go").rstrip("/")
OP_KEY = os.environ.get("OP_KEY", "")
BRIDGE_TOKEN = os.environ.get("BRIDGE_TOKEN", "")
MODEL_MAP = json.loads(os.environ.get("MODEL_MAP", "{}"))
MODELS = [m.strip() for m in os.environ.get("OP_MODELS", "muse-spark-1.2-contributor").split(",") if m.strip()]
PORT = int(os.environ.get("PORT", "8000"))
REASONING = os.environ.get("REASONING_EFFORT", "").strip()

UPSTREAM_URL = OP_BASE + "/v1/responses"
TIMEOUT = httpx.Timeout(connect=15.0, read=600.0, write=60.0, pool=60.0)

TEXT_PART_TYPES = {"text", "input_text", "output_text"}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, file=sys.stderr, flush=True)


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
                # anthropic-style tool result block smuggled in content
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


def map_tools(tools):
    out = []
    for t in tools or []:
        if not isinstance(t, dict):
            continue
        if t.get("type") != "function":
            continue
        f = t.get("function")
        if isinstance(f, dict):
            nt = {"type": "function", "name": f.get("name", ""),
                  "parameters": f.get("parameters") or {"type": "object", "properties": {}}}
            if f.get("description"):
                nt["description"] = f["description"]
            out.append(nt)
        elif t.get("name"):
            out.append(t)
    return out


def map_tool_choice(tc):
    if tc is None or isinstance(tc, str):
        return tc
    if isinstance(tc, dict) and tc.get("type") == "function":
        f = tc.get("function") or {}
        name = f.get("name") or tc.get("name")
        return {"type": "function", "name": name} if name else None
    return None
