#!/usr/bin/env python3
"""muse-bridge: protocol adapter between coding agents and OpenCode Zen.

Three inbound wires, one upstream:
1. POST /v1/chat/completions   OpenAI Chat wire            -> translate -> /v1/responses
2. POST /v1/messages           Anthropic wire (via new-api)-> same as (1)
3. POST [/v1]/responses        native Responses passthrough with safety rails

Upstream auth always comes from OP_KEY; inbound Authorization is ignored unless
BRIDGE_TOKEN is set.
"""
import hashlib
import json
import os
import re
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx

OP_BASE = os.environ.get("OP_BASE", "https://opencode.ai/zen/go").rstrip("/")
OP_KEY = os.environ.get("OP_KEY", "")
BRIDGE_TOKEN = os.environ.get("BRIDGE_TOKEN", "")
MODEL_MAP = json.loads(os.environ.get("MODEL_MAP", "{}"))
MODELS = [m.strip() for m in os.environ.get("OP_MODELS", "muse-spark-1.2-contributor").split(",") if m.strip()]
PORT = int(os.environ.get("PORT", "8000"))
REASONING_DEFAULT = os.environ.get("REASONING_EFFORT", "").strip()

UPSTREAM_URL = OP_BASE + "/v1/responses"
TIMEOUT = httpx.Timeout(connect=15.0, read=600.0, write=60.0, pool=60.0)

TEXT_PART_TYPES = {"text", "input_text", "output_text"}
THINK_SUFFIXES = (("-high", "high"), ("-medium", "medium"), ("-low", "low"))
EFFORT_ALIASES = {"minimal": "low", "max": "high", "maximum": "high", "off": "low"}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, file=sys.stderr, flush=True)


def resolve_effort(body, forced=None):
    """Precedence: model-name suffix > explicit request hint >
    thinking.budget_tokens > REASONING_EFFORT env default."""
    eff = None
    r = body.get("reasoning")
    if isinstance(r, dict):
        v = r.get("effort") or r.get("reasoningEffort")
        if v:
            eff = str(v).strip().lower()
    elif isinstance(r, str) and r.strip():
        eff = r.strip().lower()
    elif body.get("reasoning_effort"):
        eff = str(body["reasoning_effort"]).strip().lower()
    elif body.get("reasoningEffort"):
        eff = str(body["reasoningEffort"]).strip().lower()
    if not eff:
        th = body.get("thinking")
        if isinstance(th, dict) and th.get("type") == "enabled":
            try:
                b = int(th.get("budget_tokens") or 0)
            except Exception:
                b = 0
            if b:
                eff = "high" if b >= 16000 else ("medium" if b >= 4000 else "low")
    if not eff and forced:
        eff = forced
    if eff:
        eff = EFFORT_ALIASES.get(eff, eff)
    return eff if eff in ("low", "medium", "high") else (REASONING_DEFAULT or None)


def reasoning_block(effort):
    # muse only streams a visible thinking trace when a summary mode is requested
    return {"effort": effort, "summary": "auto"}
