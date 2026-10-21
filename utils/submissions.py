"""Save each student submission as its own row.

The previous app deleted the existing ``student_inputs`` row (and its
``student_sentences``) for the same student and week, then inserted a
replacement. A second submit therefore destroyed the first one.

Attempts are numbered per ``(student_id, week_id)``:

* the earliest surviving row is attempt 1 (the migration backfill),
* the next submit for that student and week is attempt 2, and so on.

Nothing in this module issues DELETE, UPDATE, REPLACE, or
``ON DUPLICATE KEY UPDATE`` against submission rows. A duplicate
``(student_id, week_id, attempt_number)`` is rejected by the database
after the migration; the insert fails and the caller rolls back.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence


class SubmissionWriteError(RuntimeError):
    """Raised when a submission write would change an existing row."""


_INSERT_INPUT = """
INSERT INTO student_inputs
  (student_id, week_id,
   student_name, email, title, story,
   total_sentences, show_sentences, tell_sentences, sentence_fragment,
   agree_fragment, disagree_fragment,
   reflection, helpfulness, comments, Class,
   attempt_number, submitted_at)
VALUES (%s, %s, %s, %s, %s, %s,
        %s, %s, %s, %s,
        %s, %s,
        %s, %s, %s, %s,
        %s, %s)
"""

_LOCK_ATTEMPTS = """
SELECT input_id, attempt_number
FROM student_inputs
WHERE student_id = %s AND week_id = %s
FOR UPDATE
"""

_INSERT_SENTENCES = """
INSERT INTO student_sentences
  (input_id, week_id, sentence_idx, sentence_text, model_label,
   student_agree, highlight_words, explanation)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
"""


def submission_total(show: int, tell: int, sentence_fragment: int) -> int:
    """Match the ``student_inputs`` check constraint (total = show + tell + fragment)."""
    return int(show) + int(tell) + int(sentence_fragment)


def next_attempt_number(existing_attempt_numbers: Iterable[Any]) -> int:
    """Return the next attempt for one student and week.

    Existing numbers are left untouched. ``None`` (a row written before
    the attempt column was backfilled) does not count as a used number.
    """
    used = []
    for value in existing_attempt_numbers:
        if isinstance(value, bool) or not isinstance(value, int):
            continue
        if value > 0:
            used.append(value)
    return (max(used) if used else 0) + 1


def _statement_head(sql: str) -> str:
    return " ".join(sql.split()).lstrip().upper()


def _reject_destructive(sql: str) -> None:
    head = _statement_head(sql)
    if head.startswith(("DELETE", "UPDATE", "REPLACE", "TRUNCATE")):
        raise SubmissionWriteError(
            "Refusing to delete or overwrite an existing submission."
        )
    if "ON DUPLICATE KEY" in head:
        raise SubmissionWriteError(
            "Refusing to upsert a submission over an existing row."
        )


def _execute(cursor: Any, sql: str, params: Sequence[Any] | None = None) -> None:
    _reject_destructive(sql)
    if params is None:
        cursor.execute(sql)
    else:
        cursor.execute(sql, params)


def _attempt_number_from_row(row: Any) -> Any:
    if isinstance(row, Mapping):
        return row.get("attempt_number")
    return row[1]


def utc_now_naive() -> datetime:
    """UTC timestamp stored in ``submitted_at`` (MySQL DATETIME has no zone)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def save_submission(
    cursor: Any,
    *,
    student_id: int,
    week_id: int,
    student_name: str,
    email: str,
    title: str,
    story: str,
    show: int,
    tell: int,
    sentence_fragment: int,
    agree_fragment: int,
    disagree_fragment: int,
    reflection: str | None,
    helpfulness: str | None,
    comments: str | None,
    student_class: str | None,
    sentence_rows: Sequence[Sequence[Any]],
    submitted_at: datetime | None = None,
) -> dict[str, Any]:
    """Insert a new attempt and its sentences.

    Locks the student's existing rows for this week so two overlapping
    submits observe each other's attempt numbers. Does not modify those
    rows. Returns ``{"input_id", "attempt_number"}``.
    """
    _execute(cursor, _LOCK_ATTEMPTS, (student_id, week_id))
    existing = list(cursor.fetchall() or [])
    attempt_number = next_attempt_number(
        _attempt_number_from_row(row) for row in existing
    )
    when = submitted_at if submitted_at is not None else utc_now_naive()
    total = submission_total(show, tell, sentence_fragment)

    _execute(
        cursor,
        _INSERT_INPUT,
        (
            student_id,
            week_id,
            student_name,
            email,
            title,
            story,
            total,
            int(show),
            int(tell),
            int(sentence_fragment),
            int(agree_fragment),
            int(disagree_fragment),
            reflection,
            helpfulness,
            comments,
            student_class,
            attempt_number,
            when,
        ),
    )
    input_id = cursor.lastrowid

    sentence_params = [
        (input_id, week_id, idx, text, label, agree, highlights, explanation)
        for (idx, text, label, agree, highlights, explanation) in sentence_rows
    ]
    if sentence_params:
        _reject_destructive(_INSERT_SENTENCES)
        cursor.executemany(_INSERT_SENTENCES, sentence_params)

    return {"input_id": input_id, "attempt_number": attempt_number}


def annotate_attempts(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Label every attempt without changing the input rows.

    This is the same rule as the ``student_attempts`` view:

    * group by ``(student_id, week_id)``
    * ``attempt_rank`` is the stored ``attempt_number`` (1 = earliest)
    * ``is_latest`` is the highest attempt number, breaking ties on the
      highest ``input_id``

    Dashboards that used to assume one row per student and week should
    read the latest row. History views should list every annotated row.
    """
    grouped: dict[tuple[Any, Any], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (row["student_id"], row["week_id"])
        grouped.setdefault(key, []).append(row)

    annotated: list[dict[str, Any]] = []
    for group in grouped.values():
        latest = max(group, key=_latest_sort_key)
        ordered = sorted(group, key=_attempt_sort_key)
        for position, row in enumerate(ordered, start=1):
            copy = dict(row)
            stored = row.get("attempt_number")
            copy["attempt_rank"] = stored if isinstance(stored, int) and not isinstance(stored, bool) else position
            copy["is_latest"] = row["input_id"] == latest["input_id"]
            annotated.append(copy)

    annotated.sort(
        key=lambda row: (
            row["student_id"],
            row["week_id"],
            row["attempt_rank"],
            row["input_id"],
        )
    )
    return annotated


def latest_attempts(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One row per student and week: the newest attempt."""
    return [row for row in annotate_attempts(rows) if row["is_latest"]]


def _latest_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    attempt = row.get("attempt_number")
    has_attempt = isinstance(attempt, int) and not isinstance(attempt, bool)
    return (1 if has_attempt else 0, attempt if has_attempt else -1, row["input_id"])


def _attempt_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    attempt = row.get("attempt_number")
    has_attempt = isinstance(attempt, int) and not isinstance(attempt, bool)
    return (0 if has_attempt else 1, attempt if has_attempt else 0, row["input_id"])
