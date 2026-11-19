"""
Assignment 11 — Audit Log.

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.

Thiết kế: observer độc lập framework (không phải ADK plugin). Pipeline gọi
``record_input`` trước lớp đầu tiên và ``record_output`` sau lớp cuối cùng,
nên log ghi được cả request bị chặn ở rate limiter / input guardrail.
Secret / PII trong log được che (log cũng là một kênh rò rỉ).
"""
from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

_MAX_LOG_TEXT = 1000


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


def _sanitize(text: str) -> str:
    """Che secret/PII trước khi ghi log + cắt độ dài."""
    text = text or ""
    try:
        from guardrails.output_guardrails import content_filter

        text = content_filter(text)["redacted"]
    except Exception:  # audit không được làm hỏng pipeline
        pass
    if len(text) > _MAX_LOG_TEXT:
        text = text[:_MAX_LOG_TEXT] + f"... [truncated {len(text) - _MAX_LOG_TEXT} chars]"
    return text


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, float] = {}
        self._pending: dict[str, dict] = {}

    @staticmethod
    def _key(user_id: str, request_id: str | None) -> str:
        return request_id or f"user:{user_id}"

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Store input + start timestamp keyed by request_id (fallback: user_id)."""
        request_id = request_id or uuid.uuid4().hex[:12]
        key = self._key(user_id, request_id)
        self._open[key] = time.perf_counter()
        self._pending[key] = {
            "request_id": request_id,
            "user_id": user_id,
            "timestamp": utc_now_iso(),
            "input": _sanitize(text),
            "input_chars": len(text or ""),
        }
        return request_id

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Store output, layer decision, latency; append to self.logs."""
        key = self._key(user_id, request_id)
        started = self._open.pop(key, None)
        entry = self._pending.pop(key, None) or {
            "request_id": request_id,
            "user_id": user_id,
            "timestamp": utc_now_iso(),
            "input": None,
            "input_chars": 0,
        }
        latency_ms = (
            round((time.perf_counter() - started) * 1000, 2) if started is not None else None
        )
        entry.update(
            {
                "output": _sanitize(text),
                "blocked": bool(blocked),
                "layer": layer,
                "latency_ms": latency_ms,
                "completed_at": utc_now_iso(),
            }
        )
        self.logs.append(entry)
        return entry

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.logs, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return str(path)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
