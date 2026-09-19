"""Dependency-light stage outcome capture for Composer observability.

The helpers in this module are intentionally optional: when no observation
queue is installed they are no-ops.  This lets the same pipeline code serve
ordinary library callers without allocating trace state or changing failure
semantics.
"""

from __future__ import annotations

import re
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Iterator, List, Optional


_local = threading.local()
_BEARER_RE = re.compile(r"(?i)(bearer\s+)[^\s,;]+")
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)(api[_-]?key|authorization|password|secret|token)"
    r"(\s*[:=]\s*)['\"]?[^\s,'\";}]+"
)
_SECRET_KEY_PARTS = ("key", "token", "secret", "password", "authorization")


@dataclass
class ObservationContext:
    queue: Any = None
    trial_id: Optional[int] = None
    sample_index: Optional[int] = None
    attempt: Optional[int] = None
    secrets: List[str] = field(default_factory=list)


def _current_context() -> ObservationContext:
    return getattr(_local, "observation_context", ObservationContext())


@contextmanager
def observation_context(
    queue: Any,
    *,
    trial_id: Optional[int] = None,
    sample_index: Optional[int] = None,
    attempt: Optional[int] = None,
    secrets: Optional[Iterable[Optional[str]]] = None,
) -> Iterator[None]:
    """Install queue and attribution metadata for the current thread."""
    previous = _current_context()
    merged = ObservationContext(
        queue=queue if queue is not None else previous.queue,
        trial_id=trial_id if trial_id is not None else previous.trial_id,
        sample_index=sample_index if sample_index is not None else previous.sample_index,
        attempt=attempt if attempt is not None else previous.attempt,
        secrets=list(previous.secrets),
    )
    merged.secrets.extend(str(value) for value in (secrets or ()) if value)
    _local.observation_context = merged
    try:
        yield
    finally:
        _local.observation_context = previous


def sanitize_error_message(value: Any, secrets: Optional[Iterable[str]] = None) -> str:
    """Return a bounded diagnostic message with common credential shapes removed."""
    text = str(value or "")
    for secret in secrets or ():
        if secret and len(secret) >= 4:
            text = text.replace(secret, "***REDACTED***")
    text = _BEARER_RE.sub(r"\1***REDACTED***", text)
    text = _SECRET_ASSIGNMENT_RE.sub(r"\1\2***REDACTED***", text)
    return text[:500]


def provider_identity(provider: Any) -> tuple[Optional[str], Optional[str]]:
    """Extract safe provider/model labels from a provider facade or wrapper."""
    if provider is None:
        return None, None
    value = getattr(provider, "provider", None)
    provider_name = getattr(value, "value", value)
    model = getattr(provider, "model", None) or getattr(provider, "model_name", None)
    return (
        str(provider_name) if provider_name is not None else type(provider).__name__,
        str(model) if model is not None else None,
    )


def collect_secret_values(value: Any, key_name: str = "") -> List[str]:
    """Recursively collect configured secret values without retaining structure."""
    if isinstance(value, dict):
        result: List[str] = []
        for key, item in value.items():
            result.extend(collect_secret_values(item, str(key)))
        return result
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            result.extend(collect_secret_values(item, key_name))
        return result
    if value and any(part in key_name.lower() for part in _SECRET_KEY_PARTS):
        return [str(value)]
    return []


def _exception_chain(exc: BaseException) -> List[BaseException]:
    chain: List[BaseException] = []
    seen = set()
    current: Optional[BaseException] = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        current = current.__cause__ or current.__context__
    return chain


def _error_details(exc: BaseException, component: str) -> Dict[str, Any]:
    chain = _exception_chain(exc)
    # Prefer the deepest typed error. Wrapper errors often replace a provider
    # exception with a less specific pipeline exception.
    typed = next(
        (item for item in reversed(chain) if getattr(item, "error_code", None)),
        None,
    )
    deepest = chain[-1]
    selected = typed or deepest
    error_code = getattr(selected, "error_code", None)
    if component == "llm":
        # A pipeline wrapper such as GenerationError may hide a recognizable
        # SDK timeout/rate-limit exception in its cause. Observability can keep
        # the more useful provider classification without changing what is
        # raised or whether Composer retries it.
        name = type(deepest).__name__.lower()
        provider_code = None
        if "timeout" in name:
            provider_code = "PROVIDER_TIMEOUT"
        elif "ratelimit" in name or "toomanyrequests" in name:
            provider_code = "PROVIDER_RATE_LIMITED"
        elif "authentication" in name or "unauthorized" in name:
            provider_code = "PROVIDER_AUTH_FAILED"
        elif "connection" in name or "unavailable" in name:
            provider_code = "PROVIDER_UNAVAILABLE"
        if provider_code is not None:
            error_code = provider_code
            selected = deepest
    if error_code is None:
        name = type(selected).__name__.lower()
        error_code = error_code or type(selected).__name__
    return {
        "error_code": str(error_code),
        "error_type": type(selected).__name__,
        "retryable": bool(getattr(selected, "retryable", False)),
        "message_source": selected,
    }


def _emit(record: Dict[str, Any]) -> None:
    context = _current_context()
    if context.queue is None:
        return
    payload = dict(record)
    payload.setdefault("event_id", str(uuid.uuid4()))
    payload.setdefault("occurred_at", datetime.now(timezone.utc).isoformat())
    payload.setdefault("trial_id", context.trial_id)
    payload.setdefault("sample_index", context.sample_index)
    payload.setdefault("attempt", context.attempt)
    if payload.get("message") is not None:
        payload["message"] = sanitize_error_message(payload["message"], context.secrets)
    context.queue.put(("operation", payload))


def emit_stage_outcome(
    stage: str,
    component: str,
    *,
    outcome: str = "success",
    duration_ms: Optional[float] = None,
    recovery: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    error: Optional[BaseException] = None,
) -> None:
    """Emit one completed stage attempt to the active trace queue."""
    record: Dict[str, Any] = {
        "stage": stage,
        "component": component,
        "outcome": outcome,
        "duration_ms": duration_ms,
        "recovery": recovery,
        "provider": provider,
        "model": model,
    }
    if error is not None:
        details = _error_details(error, component)
        record.update({
            "error_code": details["error_code"],
            "error_type": details["error_type"],
            "retryable": details["retryable"],
            "message": details["message_source"],
        })
    _emit(record)


class StageObservation:
    """Context manager that emits exactly one outcome for a stage attempt."""

    def __init__(
        self,
        stage: str,
        component: str,
        *,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        defer_propagated_error: bool = True,
    ) -> None:
        self.stage = stage
        self.component = component
        self.provider = provider
        self.model = model
        self.defer_propagated_error = defer_propagated_error
        self.started = 0.0
        self.recovered_error: Optional[BaseException] = None
        self.recovery: Optional[str] = None

    def __enter__(self) -> "StageObservation":
        self.started = time.perf_counter()
        stack = list(getattr(_local, "observation_stack", []))
        stack.append(self)
        _local.observation_stack = stack
        return self

    def mark_error(self, error: BaseException, recovery: str = "fallback") -> None:
        self.recovered_error = error
        self.recovery = recovery

    def __exit__(self, exc_type: Any, exc: Optional[BaseException], tb: Any) -> bool:
        stack = list(getattr(_local, "observation_stack", []))
        if stack and stack[-1] is self:
            stack.pop()
            _local.observation_stack = stack
        duration_ms = (time.perf_counter() - self.started) * 1000.0
        if exc is not None:
            metadata = {
                "stage": self.stage,
                "component": self.component,
                "provider": self.provider,
                "model": self.model,
                "duration_ms": duration_ms,
            }
            if not hasattr(exc, "_muffakir_observation"):
                try:
                    setattr(exc, "_muffakir_observation", metadata)
                except Exception:
                    pass
            if not self.defer_propagated_error:
                emit_stage_outcome(
                    self.stage,
                    self.component,
                    outcome="error",
                    duration_ms=duration_ms,
                    recovery="unrecovered",
                    provider=self.provider,
                    model=self.model,
                    error=exc,
                )
                try:
                    setattr(exc, "_muffakir_observation_emitted", True)
                except Exception:
                    pass
            return False
        if self.recovered_error is not None:
            emit_stage_outcome(
                self.stage,
                self.component,
                outcome="error",
                duration_ms=duration_ms,
                recovery=self.recovery or "fallback",
                provider=self.provider,
                model=self.model,
                error=self.recovered_error,
            )
        else:
            emit_stage_outcome(
                self.stage,
                self.component,
                duration_ms=duration_ms,
                provider=self.provider,
                model=self.model,
            )
        return False


def observe_stage(
    stage: str,
    component: str,
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    defer_propagated_error: bool = True,
) -> StageObservation:
    return StageObservation(
        stage,
        component,
        provider=provider,
        model=model,
        defer_propagated_error=defer_propagated_error,
    )


def mark_current_stage_error(error: BaseException, recovery: str = "fallback") -> None:
    """Mark a caught problem as the current stage's single recovered outcome."""
    stack = getattr(_local, "observation_stack", [])
    if stack:
        stack[-1].mark_error(error, recovery)


def record_exception_outcome(
    error: BaseException,
    *,
    recovery: str,
    fallback_stage: str = "trial_execution",
    fallback_component: str = "executor",
) -> None:
    """Finalize a propagated exception once retry/fatal disposition is known."""
    if any(
        bool(getattr(item, "_muffakir_observation_emitted", False))
        for item in _exception_chain(error)
    ):
        return
    metadata: Optional[Dict[str, Any]] = None
    for item in _exception_chain(error):
        candidate = getattr(item, "_muffakir_observation", None)
        if candidate:
            metadata = candidate
            break
    metadata = metadata or {}
    emit_stage_outcome(
        metadata.get("stage", fallback_stage),
        metadata.get("component", fallback_component),
        outcome="error",
        duration_ms=metadata.get("duration_ms"),
        recovery=recovery,
        provider=metadata.get("provider"),
        model=metadata.get("model"),
        error=error,
    )
    # A propagated run error can be observed by both Composer.fit() and its
    # ComposerUI wrapper. Mark the exception after the first event so the
    # error-rate denominator and fatal count stay accurate.
    try:
        setattr(error, "_muffakir_observation_emitted", True)
    except Exception:
        pass
