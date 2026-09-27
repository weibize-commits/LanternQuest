"""Auditable per-episode recording and exact-prefix replay for LLM calls."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from lanternquest.llm import LLMAdapter, LLMRequest, LLMResponse

TRACE_SCHEMA_VERSION = "lanternquest_model_trace_v2"
_ORDER_INSENSITIVE_LIST_KEYS = {
    "current_legal_actions",
    "legal_actions",
    "observed_facts",
}
_ORDER_INSENSITIVE_TEXT_KEYS = {
    "current_observation",
    "inventory",
    "look",
    "observation",
}
_VOLATILE_IDENTIFIER_KEYS = {
    "obligation_id",
    "observation_sha256",
}


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _trace_canonical_value(value: Any, *, field_name: str | None = None) -> Any:
    """Remove presentation-only ordering from ScienceWorld request hashing."""
    if field_name in _VOLATILE_IDENTIFIER_KEYS:
        return "<derived-from-presentation-order>"
    if isinstance(value, dict):
        return {
            key: _trace_canonical_value(item, field_name=str(key))
            for key, item in value.items()
        }
    if isinstance(value, list):
        normalized = [
            _trace_canonical_value(item, field_name=field_name) for item in value
        ]
        if field_name in _ORDER_INSENSITIVE_LIST_KEYS:
            return sorted(normalized, key=_canonical_json)
        return normalized
    if isinstance(value, str) and field_name in _ORDER_INSENSITIVE_TEXT_KEYS:
        return sorted(re.findall(r"[a-z0-9]+", value.casefold()))
    return value


def _request_sha256(request_payload: dict[str, Any]) -> str:
    return _sha256(_trace_canonical_value(request_payload))


class ModelTraceError(ValueError):
    """Raised when a replay request does not match the recorded prefix."""


class ModelTraceEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    call_index: int
    source: Literal["live", "replay"]
    purpose: str
    request_sha256: str
    payload_sha256: str
    request: dict[str, Any]
    response: dict[str, Any]


class ModelTraceDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = TRACE_SCHEMA_VERSION
    mode: Literal["record", "replay_prefix", "replay_until_divergence"]
    model_id: str
    profile_id: str | None = None
    source_trace_path: str | None = None
    source_trace_sha256: str | None = None
    source_call_count: int = 0
    source_consumed_count: int = 0
    prefix_complete: bool = False
    divergence_purpose: str | None = None
    mismatch_count: int = 0
    mismatches: list[dict[str, Any]] = Field(default_factory=list)
    entries: list[ModelTraceEntry] = Field(default_factory=list)


class ModelTraceAdapter:
    """Wrap an adapter to record calls or replay an exact shared prefix.

    In ``replay_prefix`` mode every ordinary request must byte-match the next
    recorded request after canonical JSON serialization. A live branch is
    permitted only for an explicitly controlled repair request after the complete
    source trace has been consumed. This makes the pre-intervention trajectory
    auditable for both D3 and D4.

    In ``replay_until_divergence`` mode matching calls are replayed until the
    first changed request. That request and every later request are delegated
    live. The planned divergence is recorded separately from replay errors.
    """

    def __init__(
        self,
        delegate: LLMAdapter,
        *,
        mode: Literal["record", "replay_prefix", "replay_until_divergence"],
    ) -> None:
        self.delegate = delegate
        self.mode = mode
        self.model_id = delegate.model_id
        self.profile_id = getattr(delegate, "profile_id", None)
        self._output_path: Path | None = None
        self._source_path: Path | None = None
        self._source_sha256: str | None = None
        self._source_entries: list[ModelTraceEntry] = []
        self._entries: list[ModelTraceEntry] = []
        self._cursor = 0
        self._divergence_purpose: str | None = None
        self._mismatch_count = 0
        self._mismatches: list[dict[str, Any]] = []
        self._active = False

    def begin_episode(
        self,
        output_path: Path,
        *,
        source_trace_path: Path | None = None,
    ) -> None:
        if self._active:
            raise ModelTraceError("A model-trace episode is already active")
        self._output_path = output_path.resolve()
        self._source_path = (
            source_trace_path.resolve() if source_trace_path is not None else None
        )
        self._source_sha256 = None
        self._source_entries = []
        self._entries = []
        self._cursor = 0
        self._divergence_purpose = None
        self._mismatch_count = 0
        self._mismatches = []
        if self.mode in {"replay_prefix", "replay_until_divergence"}:
            if self._source_path is None:
                raise ModelTraceError("replay mode requires source_trace_path")
            raw = self._source_path.read_bytes()
            self._source_sha256 = hashlib.sha256(raw).hexdigest()
            source = ModelTraceDocument.model_validate_json(raw)
            if source.model_id != self.model_id:
                raise ModelTraceError(
                    "Trace model mismatch: "
                    f"source={source.model_id!r}, adapter={self.model_id!r}"
                )
            self._source_entries = source.entries
        elif source_trace_path is not None:
            raise ModelTraceError("record mode does not accept source_trace_path")
        self._active = True

    @property
    def episode_stats(self) -> dict[str, int]:
        return {
            "model_trace_live_call_count": sum(
                entry.source == "live" for entry in self._entries
            ),
            "model_trace_replayed_call_count": sum(
                entry.source == "replay" for entry in self._entries
            ),
            "model_trace_mismatch_count": self._mismatch_count,
            "model_trace_divergence_count": int(
                self._divergence_purpose is not None
            ),
            "model_trace_source_call_count": len(self._source_entries),
            "model_trace_source_consumed_count": self._cursor,
        }

    def _mismatch(self, message: str) -> ModelTraceError:
        self._mismatch_count += 1
        return ModelTraceError(message)

    def complete(self, request: LLMRequest) -> LLMResponse:
        if not self._active:
            raise ModelTraceError("begin_episode must be called before complete")
        request_payload = request.model_dump(mode="json")
        request_hash = _request_sha256(request_payload)
        payload_hash = _sha256(
            _trace_canonical_value(request_payload["payload"])
        )

        if self.mode in {
            "replay_prefix",
            "replay_until_divergence",
        } and self._divergence_purpose is None:
            if self.mode == "replay_until_divergence":
                expected = (
                    self._source_entries[self._cursor]
                    if self._cursor < len(self._source_entries)
                    else None
                )
                if expected is None or expected.request_sha256 != request_hash:
                    self._divergence_purpose = request.purpose
                else:
                    response = LLMResponse.model_validate(expected.response)
                    self._cursor += 1
                    self._entries.append(
                        ModelTraceEntry(
                            call_index=len(self._entries) + 1,
                            source="replay",
                            purpose=request.purpose,
                            request_sha256=request_hash,
                            payload_sha256=payload_hash,
                            request=request_payload,
                            response=response.model_dump(mode="json"),
                        )
                    )
                    return response
            elif request.purpose in {"residual_repair", "candidate_ranking"}:
                if self._cursor != len(self._source_entries):
                    raise self._mismatch(
                        "Controlled repair attempted before the recorded prefix was "
                        f"fully consumed ({self._cursor}/{len(self._source_entries)})"
                    )
                self._divergence_purpose = request.purpose
            else:
                if self._cursor >= len(self._source_entries):
                    raise self._mismatch(
                        "Ordinary model call extends beyond the recorded D0 prefix: "
                        f"purpose={request.purpose!r}"
                    )
                expected = self._source_entries[self._cursor]
                if expected.request_sha256 != request_hash:
                    self._mismatches.append(
                        {
                            "call_index": self._cursor + 1,
                            "expected_request_sha256": expected.request_sha256,
                            "received_request_sha256": request_hash,
                            "expected_request": expected.request,
                            "received_request": request_payload,
                        }
                    )
                    raise self._mismatch(
                        "Recorded-prefix request mismatch at call "
                        f"{self._cursor + 1}: expected {expected.request_sha256}, "
                        f"received {request_hash}"
                    )
                response = LLMResponse.model_validate(expected.response)
                self._cursor += 1
                self._entries.append(
                    ModelTraceEntry(
                        call_index=len(self._entries) + 1,
                        source="replay",
                        purpose=request.purpose,
                        request_sha256=request_hash,
                        payload_sha256=payload_hash,
                        request=request_payload,
                        response=response.model_dump(mode="json"),
                    )
                )
                return response

        response = self.delegate.complete(request)
        self._entries.append(
            ModelTraceEntry(
                call_index=len(self._entries) + 1,
                source="live",
                purpose=request.purpose,
                request_sha256=request_hash,
                payload_sha256=payload_hash,
                request=request_payload,
                response=response.model_dump(mode="json"),
            )
        )
        return response

    def end_episode(self) -> ModelTraceDocument:
        if not self._active or self._output_path is None:
            raise ModelTraceError("No active model-trace episode")
        prefix_complete = (
            self.mode == "record"
            or self._cursor == len(self._source_entries)
        )
        document = ModelTraceDocument(
            mode=self.mode,
            model_id=self.model_id,
            profile_id=self.profile_id,
            source_trace_path=(
                str(self._source_path) if self._source_path is not None else None
            ),
            source_trace_sha256=self._source_sha256,
            source_call_count=len(self._source_entries),
            source_consumed_count=self._cursor,
            prefix_complete=prefix_complete,
            divergence_purpose=self._divergence_purpose,
            mismatch_count=self._mismatch_count,
            mismatches=self._mismatches,
            entries=self._entries,
        )
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self._output_path.with_suffix(
            self._output_path.suffix + ".tmp"
        )
        temporary_path.write_text(
            json.dumps(document.model_dump(mode="json"), ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(self._output_path)
        self._active = False
        return document
