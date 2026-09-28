from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any, Callable, Mapping


class ProviderReplayConflict(RuntimeError):
    """An idempotency key conflicts with, or is blocked by, durable provider state."""


class DurableReplayStore:
    """SQLite-backed two-phase Provider ABI replay journal.

    Safety contract:
    1. atomically claim the idempotency key as INFLIGHT and COMMIT;
    2. perform the provider operation outside the SQLite transaction;
    3. atomically persist the full response as COMPLETED.

    If the process disappears after step 1 but before step 3, the INFLIGHT row
    survives. A duplicate request is rejected rather than re-executed because
    the external side effect may already have happened.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()

    def run(
        self,
        request: Mapping[str, Any],
        operation: Callable[[], dict[str, object]],
    ) -> dict[str, object]:
        request_id = _non_empty_string(request.get("request_id"), "request_id")
        execution_key = _sha256(request.get("execution_key"), "execution_key")
        idempotency_key = _sha256(
            request.get("idempotency_key"),
            "idempotency_key",
        )
        action = _non_empty_string(request.get("action"), "action")
        request_hash = _request_hash(request)

        replay = self._claim_or_replay(
            execution_key=execution_key,
            idempotency_key=idempotency_key,
            action=action,
            request_hash=request_hash,
        )
        if replay is not None:
            replay["request_id"] = request_id
            return replay

        response = operation()
        if not isinstance(response, dict):
            raise TypeError("provider operation must return a response object")
        completed = dict(response)
        completed["execution_key"] = execution_key
        completed["idempotency_key"] = idempotency_key
        completed["replayed"] = False

        self._complete(
            execution_key=execution_key,
            idempotency_key=idempotency_key,
            action=action,
            request_hash=request_hash,
            response=completed,
        )
        return completed

    def _claim_or_replay(
        self,
        *,
        execution_key: str,
        idempotency_key: str,
        action: str,
        request_hash: str,
    ) -> dict[str, object] | None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = self._connect()
        try:
            self._create_schema(connection)
            self._quick_check(connection)
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT execution_key, action, request_sha256, status, response_json
                FROM provider_replay
                WHERE idempotency_key = ?
                """,
                (idempotency_key,),
            ).fetchone()

            if row is not None:
                self._validate_binding(
                    row,
                    execution_key=execution_key,
                    action=action,
                    request_hash=request_hash,
                )
                if row["status"] == "completed":
                    response = json.loads(row["response_json"])
                    if not isinstance(response, dict):
                        raise ValueError("corrupt provider replay response")
                    response["replayed"] = True
                    connection.execute("COMMIT")
                    return response
                if row["status"] == "inflight":
                    raise ProviderReplayConflict(
                        "prior provider operation is INFLIGHT/ambiguous; "
                        "automatic retry is blocked"
                    )
                raise ValueError(
                    f"corrupt provider replay status: {row['status']!r}"
                )

            connection.execute(
                """
                INSERT INTO provider_replay(
                    idempotency_key,
                    execution_key,
                    action,
                    request_sha256,
                    status,
                    response_json
                ) VALUES(?, ?, ?, ?, 'inflight', '{}')
                """,
                (
                    idempotency_key,
                    execution_key,
                    action,
                    request_hash,
                ),
            )
            connection.execute("COMMIT")
            return None
        except Exception:
            _rollback_quietly(connection)
            raise
        finally:
            connection.close()

    def _complete(
        self,
        *,
        execution_key: str,
        idempotency_key: str,
        action: str,
        request_hash: str,
        response: Mapping[str, object],
    ) -> None:
        connection = self._connect()
        try:
            self._create_schema(connection)
            self._quick_check(connection)
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT execution_key, action, request_sha256, status
                FROM provider_replay
                WHERE idempotency_key = ?
                """,
                (idempotency_key,),
            ).fetchone()
            if row is None:
                raise ProviderReplayConflict(
                    "provider replay claim disappeared before completion"
                )
            self._validate_binding(
                row,
                execution_key=execution_key,
                action=action,
                request_hash=request_hash,
            )
            if row["status"] != "inflight":
                raise ProviderReplayConflict(
                    "provider replay entry is no longer INFLIGHT"
                )
            connection.execute(
                """
                UPDATE provider_replay
                SET status = 'completed', response_json = ?
                WHERE idempotency_key = ? AND status = 'inflight'
                """,
                (_canonical_json(response), idempotency_key),
            )
            changed = connection.execute(
                "SELECT changes() AS changes"
            ).fetchone()["changes"]
            if changed != 1:
                raise ProviderReplayConflict(
                    "provider replay completion lost a concurrent race"
                )
            connection.execute("COMMIT")
        except Exception:
            _rollback_quietly(connection)
            raise
        finally:
            connection.close()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path,
            timeout=30.0,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _create_schema(self, connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS provider_replay(
                idempotency_key TEXT PRIMARY KEY,
                execution_key TEXT NOT NULL,
                action TEXT NOT NULL,
                request_sha256 TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'completed',
                response_json TEXT NOT NULL
            )
            """
        )
        columns = {
            row["name"]
            for row in connection.execute(
                "PRAGMA table_info(provider_replay)"
            )
        }
        if "status" not in columns:
            connection.execute(
                """
                ALTER TABLE provider_replay
                ADD COLUMN status TEXT NOT NULL DEFAULT 'completed'
                """
            )

    def _quick_check(self, connection: sqlite3.Connection) -> None:
        results = [row[0] for row in connection.execute("PRAGMA quick_check")]
        if results != ["ok"]:
            raise RuntimeError(
                "provider replay SQLite quick_check failed: "
                + "; ".join(str(item) for item in results)
            )

    @staticmethod
    def _validate_binding(
        row: sqlite3.Row,
        *,
        execution_key: str,
        action: str,
        request_hash: str,
    ) -> None:
        if (
            row["execution_key"] != execution_key
            or row["action"] != action
            or row["request_sha256"] != request_hash
        ):
            raise ProviderReplayConflict(
                "idempotency key was reused for a different provider operation"
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


def _rollback_quietly(connection: sqlite3.Connection) -> None:
    try:
        connection.execute("ROLLBACK")
    except sqlite3.Error:
        pass


def _sha256(value: object, field_name: str) -> str:
    text = _non_empty_string(value, field_name).lower()
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hex digest")
    return text


def _non_empty_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{field_name} must be a non-empty string")
    return value.strip()
