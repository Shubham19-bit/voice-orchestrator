"""Minimal async HTTP helpers. Uses httpx when installed (recommended, keeps a warm
connection pool — that matters: a cold TLS handshake can cost 100+ ms per call),
and falls back to stdlib urllib in a thread otherwise."""
from __future__ import annotations

import asyncio
import json
import urllib.request

try:
    import httpx  # type: ignore
except ImportError:  # pragma: no cover
    httpx = None

_clients: dict[float, "httpx.AsyncClient"] = {}


class HTTPError(RuntimeError):
    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {body[:300]}")
        self.status = status


def _client(timeout_s: float):
    c = _clients.get(timeout_s)
    if c is None or c.is_closed:
        c = httpx.AsyncClient(timeout=timeout_s, http2=False,
                              limits=httpx.Limits(max_keepalive_connections=10, keepalive_expiry=60))
        _clients[timeout_s] = c
    return c


async def post_json(url: str, body: dict, headers: dict, timeout_s: float = 10.0) -> dict:
    headers = {"Content-Type": "application/json", **headers}
    if httpx is not None:
        r = await _client(timeout_s).post(url, json=body, headers=headers)
        if r.status_code >= 400:
            raise HTTPError(r.status_code, r.text)
        return r.json()

    def _do():
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:  # type: ignore[attr-defined]
            raise HTTPError(e.code, e.read().decode(errors="replace")) from None

    return await asyncio.to_thread(_do)


async def aclose_all() -> None:
    for c in list(_clients.values()):
        await c.aclose()
    _clients.clear()
