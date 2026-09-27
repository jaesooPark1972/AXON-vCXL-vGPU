from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any, Callable, Mapping


class ProviderReplayConflict(RuntimeError):
    """An idempotency key was reused for a different provider operation."""


class DurableReplayStore:
    """SQLite-backed durable replay cache for Provider ABI actions."""

    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()

    def run(
        self,
        request: Mapping[str, Any],
        operation: Callable[[], dict[str, object]],
    ) -> dict[str, object]:
        execution_key = _sha256(request.get("execution_key"), "execution_key")
        idempotency_key = _non_empty_string(
            request.get("idempotency_key"),
            "idempotency_key",
        )
        action = _non_empty_string(request.get("action"), "action")
        request_hash = _request_hash(request)

        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(
            self.path,
            timeout=30.0,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 30000")
        try:
            self._create_schema(connection)
            self._quick_check(connection)
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT execution_key, action, request_sha256, response_json
                FROM provider_replay
                WHERE idempotency_key = ?
                """,
                (idempotency_key,),
            ).fetchone()
            if row is not None:
                if (
                    row["execution_key"] != execution_key
                    or row["action"] != action
                    or row["request_sha256"] != request_hash
                ):
                    raise ProviderReplayConflict(
                        "idempotency key was reused for a different operation"
                    )
                response = json.loads(row["response_json"])
                if not isinstance(response, dict):
                    raise ValueError("corrupt provider replay response")
                response["replayed"] = True
                connection.execute("COMMIT")
                return response

            response = operation()
            if not isinstance(response, dict):
                raise TypeError("provider operation must return a response object")
            response = dict(response)
            response["execution_key"] = execution_key
            response["idempotency_key"] = idempotency_key
            response["replayed"] = False
            connection.execute(
                """
                INSERT INTO provider_replay(
                    idempotency_key,
                    execution_key,
                    action,
                    request_sha256,
                    response_json
                ) VALUES(?, ?, ?, ?, ?)
                """,
                (
                    idempotency_key,
                    execution_key,
                    action,
                    request_hash,
                    _canonical_json(response),
                ),
            )
            connection.execute("COMMIT")
            return response
        except Exception:
            try:
                connection.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise
        finally:
            connection.close()

    def _create_schema(self, connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS provider_replay(
                idempotency_key TEXT PRIMARY KEY,
                execution_key TEXT NOT NULL,
                action TEXT NOT NULL,
                request_sha256 TEXT NOT NULL,
                response_json TEXT NOT NULL
            )
            """
        )

    def _quick_check(self, connection: sqlite3.Connection) -> None:
        results = [row[0] for row in connection.execute("PRAGMA quick_check")]
        if results != ["ok"]:
            raise RuntimeError(
                "provider replay SQLite quick_check failed: "
                + "; ".join(str(item) for item in results)
            )


def replay_db_for_state(state_path: str | Path) -> Path:
    path = Path(state_path).resolve()
    return path.with_suffix(path.suffix + ".replay.sqlite3")


def _request_hash(request: Mapping[str, Any]) -> str:
    canonical = {
        "execution_key": request.get("execution_key"),
        "idempotency_key": request.get("idempotency_key"),
        "action": request.get("action"),
        "payload": request.get("payload", {}),
    }
    return hashlib.sha256(
        _canonical_json(canonical).encode("utf-8")
    ).hexdigest()


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256(value: object, field_name: str) -> str:
    text = _non_empty_string(value, field_name).lower()
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hex digest")
    return text


def _non_empty_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{field_name} must be a non-empty string")
    return value.strip()
