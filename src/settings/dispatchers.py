"""
Dispatcher configuration loading (JSON file/env, with legacy env fallback).
"""

import json
import logging
import os
from pathlib import Path

_log = logging.getLogger("utils")
_PLACEHOLDER_XMPP_DOMAINS = {"your.xmpp.server"}


def _parse_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _legacy_dispatchers(domain: str) -> dict[str, dict]:
    # Ordered to match hotkey assignments (Cmd+1..4), then the rest.
    return {
        "acorn": {
            "jid": os.getenv("ACORN_JID", f"acorn@{domain}"),
            "password": os.getenv("ACORN_PASSWORD", ""),
            "engine": "external",
            "agent": None,
            "label": "Acorn",
            "direct": True,
        },
        "cc": {
            "jid": os.getenv("CC_JID", f"cc@{domain}"),
            "password": os.getenv("CC_PASSWORD", ""),
            "engine": "claude",
            "agent": None,
            "label": "Claude Code",
        },
        "pi-gpt": {
            "jid": os.getenv("PI_GPT_JID", f"pi-gpt@{domain}"),
            "password": os.getenv("PI_GPT_PASSWORD", os.getenv("XMPP_PASSWORD", "")),
            "engine": "pi",
            "agent": "bridge-gpt",
            "model_id": os.getenv("PI_GPT_MODEL_ID", "openai-codex/gpt-5.6-sol"),
            "reasoning_mode": "xhigh",
            "label": "GPT 5.6 Sol",
        },
        "oc": {
            "jid": os.getenv("OC_JID", f"oc@{domain}"),
            "password": os.getenv("OC_PASSWORD", ""),
            "engine": "pi",
            "agent": "bridge",
            "model_id": os.getenv("OC_MODEL_ID", ""),
            "label": "Qwen 122B",
        },
        "oc-glm-zen": {
            "jid": os.getenv("OC_GLM_ZEN_JID", f"oc-glm-zen@{domain}"),
            "password": os.getenv(
                "OC_GLM_ZEN_PASSWORD", os.getenv("XMPP_PASSWORD", "")
            ),
            "engine": "pi",
            "agent": "bridge-zen",
            "model_id": os.getenv("OC_GLM_ZEN_MODEL_ID", "opencode/glm-4.7"),
            "label": "GLM 4.7 Zen",
        },
        "oc-gpt-or": {
            "jid": os.getenv("OC_GPT_OR_JID", f"oc-gpt-or@{domain}"),
            "password": os.getenv("OC_GPT_OR_PASSWORD", os.getenv("XMPP_PASSWORD", "")),
            "engine": "pi",
            "agent": "bridge-gpt-or",
            "model_id": os.getenv("OC_GPT_OR_MODEL_ID", "openrouter/openai/gpt-5.2"),
            "label": "GPT 5.2 OR",
        },
        "oc-kimi-coding": {
            "jid": os.getenv("OC_KIMI_CODING_JID", f"oc-kimi-coding@{domain}"),
            "password": os.getenv(
                "OC_KIMI_CODING_PASSWORD", os.getenv("XMPP_PASSWORD", "")
            ),
            "engine": "pi",
            "agent": "bridge-kimi-coding",
            "model_id": os.getenv(
                "OC_KIMI_CODING_MODEL_ID", "kimi-for-coding/kimi-k2.6"
            ),
            "label": "Kimi K2.6 Coding",
        },
        "loom": {
            "jid": os.getenv("LOOM_JID", f"loom@{domain}"),
            "password": os.getenv("LOOM_PASSWORD", ""),
            "engine": "pi",
            "agent": "bridge",
            "model_id": os.getenv(
                "LOOM_MODEL_ID", "local-llama/glm-4.7-flash-heretic.Q8_0.gguf"
            ),
            "label": "GLM 4.7 Flash",
        },
    }


def _normalize_dispatchers(payload: object, *, domain: str) -> dict[str, dict]:
    """Normalize dispatcher config from list/dict JSON to internal mapping."""

    entries: list[tuple[str, dict]] = []
    if isinstance(payload, list):
        for i, item in enumerate(payload):
            if isinstance(item, dict):
                name = str(item.get("name") or item.get("id") or f"dispatcher-{i + 1}")
                entries.append((name, item))
    elif isinstance(payload, dict):
        for key, value in payload.items():
            if isinstance(value, dict):
                item = dict(value)
                item.setdefault("name", str(key))
                entries.append((str(key), item))
    else:
        raise ValueError("dispatchers config must be a JSON list or object")

    out: dict[str, dict] = {}
    for fallback_name, item in entries:
        name = str(item.get("name") or fallback_name).strip() or fallback_name
        jid = str(item.get("jid") or f"{name}@{domain}").strip()
        if jid:
            bare, sep, resource = jid.partition("/")
            localpart, at, jid_domain = bare.partition("@")
            if (
                at
                and jid_domain in _PLACEHOLDER_XMPP_DOMAINS
                and domain not in _PLACEHOLDER_XMPP_DOMAINS
            ):
                bare = f"{localpart}@{domain}"
                jid = bare if not sep else f"{bare}/{resource}"
        if not jid:
            _log.warning("Skipping dispatcher %s: missing jid", name)
            continue

        password = ""
        if isinstance(item.get("password"), str):
            password = item.get("password", "").strip()
        elif isinstance(item.get("password_env"), str):
            password = os.getenv(item.get("password_env", ""), "").strip()

        engine = str(item.get("engine") or "pi").strip().lower()
        agent = item.get("agent")
        if isinstance(agent, str):
            agent = agent.strip() or None
        elif agent is not None:
            agent = str(agent).strip() or None

        entry: dict[str, object] = {
            "jid": jid,
            "password": password,
            "engine": engine,
            "agent": agent,
            "label": str(item.get("label") or name),
        }

        model_id = item.get("model_id")
        if isinstance(model_id, str) and model_id.strip():
            entry["model_id"] = model_id.strip()

        base_url = item.get("base_url")
        if isinstance(base_url, str) and base_url.strip():
            entry["base_url"] = base_url.strip().rstrip("/")

        reasoning_mode = item.get("reasoning_mode")
        if isinstance(reasoning_mode, str) and reasoning_mode.strip():
            entry["reasoning_mode"] = reasoning_mode.strip().lower()

        # Canonical name is system_prompt_extra. Keep append_system_prompt as
        # a config-file compatibility alias used by early deployments.
        system_prompt_extra = item.get("system_prompt_extra")
        if system_prompt_extra is None:
            system_prompt_extra = item.get("append_system_prompt")
        if isinstance(system_prompt_extra, str) and system_prompt_extra.strip():
            entry["system_prompt_extra"] = system_prompt_extra.strip()

        if _parse_bool(item.get("direct"), default=False):
            entry["direct"] = True
        if _parse_bool(item.get("disabled"), default=False):
            entry["disabled"] = True
        if item.get("delegation_context") is False:
            entry["delegation_context"] = False

        out[name] = entry

    return out


def _load_dispatchers_config(domain: str) -> dict[str, dict]:
    """Load dispatchers from JSON env/file, with legacy fallback."""

    raw_json = (os.getenv("SWITCH_DISPATCHERS_JSON", "") or "").strip()
    raw_file = (os.getenv("SWITCH_DISPATCHERS_FILE", "") or "").strip()
    default_files = [
        Path.cwd() / "dispatchers.local.json",
        Path.cwd() / "dispatchers.json",
        Path.cwd() / "dispatchers.example.json",
    ]

    payload: object | None = None
    if raw_json:
        try:
            payload = json.loads(raw_json)
        except Exception as e:
            _log.warning(
                "Invalid SWITCH_DISPATCHERS_JSON; using legacy dispatchers: %s", e
            )
    elif raw_file:
        try:
            payload = json.loads(Path(raw_file).read_text())
        except Exception as e:
            _log.warning(
                "Invalid SWITCH_DISPATCHERS_FILE; using legacy dispatchers: %s", e
            )
    else:
        for path in default_files:
            if not path.exists():
                continue
            try:
                payload = json.loads(path.read_text())
                _log.info("Loaded dispatchers from %s", path)
                break
            except Exception as e:
                _log.warning("Invalid dispatcher config %s: %s", path, e)

    if payload is None:
        return _legacy_dispatchers(domain)

    try:
        cfg = _normalize_dispatchers(payload, domain=domain)
        if cfg:
            return cfg
        _log.warning(
            "Dispatcher config resolved to empty set; using legacy dispatchers"
        )
    except Exception as e:
        _log.warning(
            "Failed to normalize dispatchers config; using legacy dispatchers: %s", e
        )
    return _legacy_dispatchers(domain)
