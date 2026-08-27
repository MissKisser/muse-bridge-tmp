#!/usr/bin/env python3
import json
import os
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx

OP_BASE = os.environ.get("OP_BASE", "https://opencode.ai/zen/go").rstrip("/")
OP_KEY = os.environ.get("OP_KEY", "")
LISTEN = int(os.environ.get("PORT", "8000"))
UPSTREAM_URL = OP_BASE + "/v1/responses"
TIMEOUT = httpx.Timeout(connect=15.0, read=600.0, write=60.0, pool=60.0)

PLACEHOLDER_ALL_SECTIONS_BELOW