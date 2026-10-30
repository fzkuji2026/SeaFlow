"""Grok (xAI subscription) chat-completions client."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

import requests

from glee_agent.net import patch_dns

logger = logging.getLogger("glee_agent.llm")

DEFAULT_MODEL = "grok-4.6"
AUTH_PATH = Path.home() / ".openprogram" / "auth" / "xai-subscription" / "default.json"
_COOLDOWN_UNTIL = 0.0
_LOCK = threading.Lock()
_TOKEN = ""
_TOKEN_EXP = 0.0


class LLMError(RuntimeError):
    pass


def cooling_down() -> bool:
    return time.monotonic() < _COOLDOWN_UNTIL


def _trip(seconds: float) -> None:
    global _COOLDOWN_UNTIL
    _COOLDOWN_UNTIL = max(_COOLDOWN_UNTIL, time.monotonic() + seconds)


def _redact(text: str) -> str:
    for secret in (_TOKEN, os.environ.get("XAI_API_KEY") or "", os.environ.get("GROK_API_KEY") or ""):
        if secret and secret in text:
            text = text.replace(secret, "REDACTED")
    return text


def _load_oauth() -> dict[str, Any]:
    data = json.loads(AUTH_PATH.read_text())
    creds = data.get("credentials") or []
    if not creds:
        raise LLMError("no xai-subscription credentials")
    payload = creds[0].get("payload") or {}
    inner = payload.get("data") or {}
    return {
        "access": str(payload.get("auth_value") or ""),
        "refresh": str(inner.get("refresh_token") or ""),
        "expires_at_ms": float(inner.get("expires_at_ms") or 0),
        "client_id": str(inner.get("client_id") or ""),
        "token_endpoint": str(inner.get("token_endpoint") or "https://auth.x.ai/oauth2/token"),
        "raw": data,
    }


def _save_oauth(raw: dict[str, Any], access: str, refresh: str, expires_at_ms: float) -> None:
    creds = raw.get("credentials") or []
    if not creds:
        return
    payload = creds[0].setdefault("payload", {})
    payload["auth_value"] = access
    inner = payload.setdefault("data", {})
    inner["refresh_token"] = refresh
    inner["expires_at_ms"] = int(expires_at_ms)
    creds[0]["updated_at_ms"] = int(time.time() * 1000)
    tmp = AUTH_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(raw, indent=2) + "\n")
    tmp.replace(AUTH_PATH)


def _refresh(oauth: dict[str, Any]) -> None:
    global _TOKEN, _TOKEN_EXP
    patch_dns()
    resp = requests.post(
        oauth["token_endpoint"],
        data={
            "grant_type": "refresh_token",
            "refresh_token": oauth["refresh"],
            "client_id": oauth["client_id"],
        },
        timeout=20,
    )
    if resp.status_code >= 400:
        raise LLMError(_redact(f"xai refresh {resp.status_code}: {resp.text[:180]}"))
    body = resp.json()
    access = str(body.get("access_token") or "")
    if not access:
        raise LLMError("xai refresh returned no access_token")
    refresh = str(body.get("refresh_token") or oauth["refresh"])
    expires_in = float(body.get("expires_in") or 3600)
    expires_at_ms = time.time() * 1000 + expires_in * 1000
    _TOKEN = access
    _TOKEN_EXP = time.time() + expires_in - 120
    try:
        _save_oauth(oauth["raw"], access, refresh, expires_at_ms)
    except OSError as exc:
        logger.warning("could not persist refreshed grok token: %s", exc)


def _bearer() -> str:
    global _TOKEN, _TOKEN_EXP
    env = os.environ.get("XAI_API_KEY") or os.environ.get("GROK_API_KEY") or ""
    if env:
        return env
    with _LOCK:
        now = time.time()
        if _TOKEN and now < _TOKEN_EXP:
            return _TOKEN
        oauth = _load_oauth()
        if oauth["access"] and oauth["expires_at_ms"] / 1000 - 120 > now:
            _TOKEN = oauth["access"]
            _TOKEN_EXP = oauth["expires_at_ms"] / 1000 - 120
            return _TOKEN
        _refresh(oauth)
        return _TOKEN


def _extract_text(body: dict[str, Any]) -> str:
    choices = body.get("choices") or []
    if not choices:
        raise LLMError(body.get("error") or "no choices")
    msg = (choices[0].get("message") or {})
    text = str(msg.get("content") or "").strip()
    if not text:
        raise LLMError("empty model text")
    return text


def complete(
    user: str,
    system: str = "",
    timeout: float = 45.0,
    temperature: float = 0.2,
    max_tokens: int = 512,
) -> str:
    if cooling_down():
        raise LLMError("llm cooldown")
    patch_dns()
    token = _bearer()
    model = os.environ.get("GLEE_LLM_MODEL") or DEFAULT_MODEL
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    try:
        resp = requests.post(
            "https://api.x.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json=payload,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        _trip(20)
        raise LLMError(_redact(str(exc))) from exc
    if resp.status_code in (401, 403):
        with _LOCK:
            global _TOKEN, _TOKEN_EXP
            _TOKEN, _TOKEN_EXP = "", 0.0
        _trip(5)
        raise LLMError(f"xai {resp.status_code} auth")
    if resp.status_code == 429:
        _trip(60)
        raise LLMError("xai 429 quota")
    if resp.status_code >= 400:
        raise LLMError(_redact(f"xai {resp.status_code}: {resp.text[:180]}"))
    return _extract_text(resp.json())


def parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text[: text.rfind("```")]
        text = text.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise
        data = json.loads(text[start : end + 1])
    if not isinstance(data, dict):
        raise LLMError("model JSON was not an object")
    return data
