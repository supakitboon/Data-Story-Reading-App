"""Submission attempts are inserted, never overwritten."""

from __future__ import annotations

import re
import unittest
from datetime import datetime
from pathlib import Path

from utils.submissions import (
    SubmissionWriteError,
    annotate_attempts,
    latest_attempts,
    next_attempt_number,
    save_submission,
    submission_total,
)


ROOT = Path(__file__).resolve().parents[1]
MIGRATION = ROOT / "migrations" / "2026_10_21_keep_submission_attempts.sql"


class MemoryCursor:
    """Just enough of a DB-API cursor to record submission writes."""

    def __init__(self):
        self.statements = []
        self.inputs = []
        self.sentences = []
        self.lastrowid = None
        self._next_id = 1
        self._pending_rows = []

    def execute(self, sql, params=None):
        text = " ".join(sql.split())
        self.statements.append((text, params))
        head = text.lstrip().upper()
        if head.startswith("SELECT"):
            student_id, week_id = params
            self._pending_rows = [
                (row["input_id"], row["attempt_number"])
                for row in self.inputs
                if row["student_id"] == student_id and row["week_id"] == week_id
            ]
            return
        if head.startswith("INSERT INTO STUDENT_INPUTS"):
            (
                student_id,
                week_id,
                student_name,
                email,
                title,
                story,
                total,
                show,
                tell,
                sentence_fragment,
                agree_fragment,
                disagree_fragment,
                reflection,
                helpfulness,
                comments,
                student_class,
                attempt_number,
            ) = params
            row = {
                "input_id": self._next_id,
                "student_id": student_id,
                "week_id": week_id,
                "student_name": student_name,
                "email": email,
                "title": title,
                "story": story,
                "total_sentences": total,
                "show_sentences": show,
                "tell_sentences": tell,
                "sentence_fragment": sentence_fragment,
                "agree_fragment": agree_fragment,
                "disagree_fragment": disagree_fragment,
                "reflection": reflection,
                "helpfulness": helpfulness,
                "comments": comments,
                "Class": student_class,
                "attempt_number": attempt_number,
            }
            self._next_id += 1
            self.inputs.append(row)
            self.lastrowid = row["input_id"]
            return
        raise AssertionError(f"unexpected SQL: {text}")

    def executemany(self, sql, seq):
        text = " ".join(sql.split())
        head = text.lstrip().upper()
        if not head.startswith("INSERT INTO STUDENT_SENTENCES"):
            raise AssertionError(f"unexpected executemany: {text}")
        for params in seq:
            self.statements.append((text, params))
            self.sentences.append(params)

    def fetchall(self):
        return list(self._pending_rows)


def _save(cursor, **overrides):
    payload = dict(
        student_id=7,
        week_id=3,
        student_name="Ada",
        email="ada@school.edu",
        title="First title",
        story="The chart shows 10.",
        show=1,
        tell=0,
        sentence_fragment=0,
        agree_fragment=0,
        disagree_fragment=0,
        reflection="I looked again.",
        helpfulness="yes",
        comments="",
        student_class="Karen",
        sentence_rows=[(0, "The chart shows 10.", "Show", 1, "shows", "visible")],
    )
    payload.update(overrides)
    return save_submission(cursor, **payload)


class SaveSubmissionTests(unittest.TestCase):
    def test_second_submit_inserts_a_new_row_and_keeps_the_first(self):
        cursor = MemoryCursor()
        first = _save(cursor, story="Original story.", title="Original")
        second = _save(
            cursor,
            story="Revised story.",
            title="Revised",
            show=0,
            tell=1,
            sentence_rows=[(0, "Revised story.", "Tell", 0, "because", "claim")],
        )

        self.assertEqual(first["attempt_number"], 1)
        self.assertEqual(second["attempt_number"], 2)
        self.assertNotEqual(first["input_id"], second["input_id"])
        self.assertEqual(len(cursor.inputs), 2)
        self.assertEqual(cursor.inputs[0]["story"], "Original story.")
        self.assertEqual(cursor.inputs[0]["title"], "Original")
        self.assertEqual(cursor.inputs[0]["attempt_number"], 1)
        self.assertEqual(cursor.inputs[1]["story"], "Revised story.")
        self.assertEqual(len(cursor.sentences), 2)
        self.assertEqual(cursor.sentences[0][0], first["input_id"])
        self.assertEqual(cursor.sentences[1][0], second["input_id"])

        destructive = [
            sql
            for sql, _ in cursor.statements
            if sql.lstrip().upper().startswith(("DELETE", "UPDATE", "REPLACE", "TRUNCATE"))
        ]
        self.assertEqual(destructive, [])
        input_inserts = [
            sql
            for sql, _ in cursor.statements
            if sql.lstrip().upper().startswith("INSERT INTO STUDENT_INPUTS")
        ]
        self.assertTrue(input_inserts)
        self.assertNotIn("submitted_at", input_inserts[0].lower())

    def test_attempt_numbers_restart_each_week_and_each_student(self):
        cursor = MemoryCursor()
        week_three = _save(cursor, week_id=3)
        week_four = _save(cursor, week_id=4, story="Other week")
        other_student = _save(cursor, student_id=8, week_id=3, story="Other student")

        self.assertEqual(week_three["attempt_number"], 1)
        self.assertEqual(week_four["attempt_number"], 1)
        self.assertEqual(other_student["attempt_number"], 1)
        self.assertEqual(len(cursor.inputs), 3)

    def test_existing_attempt_is_not_renumbered(self):
        cursor = MemoryCursor()
        cursor.inputs.append(
            {
                "input_id": 40,
                "student_id": 7,
                "week_id": 3,
                "attempt_number": 1,
                "story": "Already stored",
                "title": "Kept",
            }
        )
        cursor._next_id = 41
        saved = _save(cursor, story="New attempt")

        self.assertEqual(saved["attempt_number"], 2)
        self.assertEqual(cursor.inputs[0]["story"], "Already stored")
        self.assertEqual(cursor.inputs[0]["attempt_number"], 1)
        self.assertEqual(cursor.inputs[0]["input_id"], 40)

    def test_total_follows_the_check_constraint_not_a_stale_argument(self):
        self.assertEqual(submission_total(2, 3, 4), 9)
        cursor = MemoryCursor()
        _save(cursor, show=2, tell=3, sentence_fragment=4)
        self.assertEqual(cursor.inputs[0]["total_sentences"], 9)

    def test_next_attempt_number_ignores_nulls_from_before_backfill(self):
        self.assertEqual(next_attempt_number([]), 1)
        self.assertEqual(next_attempt_number([None, None]), 1)
        self.assertEqual(next_attempt_number([1, 2]), 3)

    def test_destructive_sql_is_refused(self):
        from utils.submissions import _execute

        with self.assertRaises(SubmissionWriteError):
            _execute(MemoryCursor(), "DELETE FROM student_inputs WHERE input_id = %s", (1,))
        with self.assertRaises(SubmissionWriteError):
            _execute(
                MemoryCursor(),
                "INSERT INTO student_inputs (student_id) VALUES (%s) ON DUPLICATE KEY UPDATE title = %s",
                (1, "x"),
            )


class ReadAttemptTests(unittest.TestCase):
    def test_latest_is_the_highest_attempt_and_history_lists_all(self):
        rows = [
            {
                "input_id": 10,
                "student_id": 1,
                "week_id": 3,
                "attempt_number": 1,
                "submitted_at": None,
                "story": "first",
            },
            {
                "input_id": 11,
                "student_id": 1,
                "week_id": 3,
                "attempt_number": 2,
                "submitted_at": datetime(2026, 10, 21, 13, 0, 0),
                "story": "second",
            },
            {
                "input_id": 12,
                "student_id": 1,
                "week_id": 4,
                "attempt_number": 1,
                "submitted_at": datetime(2026, 10, 21, 14, 0, 0),
                "story": "other week",
            },
            {
                "input_id": 13,
                "student_id": 2,
                "week_id": 3,
                "attempt_number": 1,
                "submitted_at": None,
                "story": "other student",
            },
        ]
        annotated = annotate_attempts(rows)
        latest = latest_attempts(rows)

        self.assertEqual([row["story"] for row in annotated if row["student_id"] == 1 and row["week_id"] == 3], ["first", "second"])
        self.assertEqual(
            {(row["student_id"], row["week_id"], row["story"]) for row in latest},
            {(1, 3, "second"), (1, 4, "other week"), (2, 3, "other student")},
        )
        self.assertFalse(annotated[0]["is_latest"])
        self.assertTrue(next(row["is_latest"] for row in annotated if row["input_id"] == 11))
        self.assertEqual(rows[0]["story"], "first")
        self.assertNotIn("is_latest", rows[0])

    def test_tie_on_attempt_number_uses_the_higher_input_id(self):
        rows = [
            {"input_id": 2, "student_id": 1, "week_id": 1, "attempt_number": 1, "story": "lower id"},
            {"input_id": 5, "student_id": 1, "week_id": 1, "attempt_number": 1, "story": "higher id"},
        ]
        latest = latest_attempts(rows)
        self.assertEqual(len(latest), 1)
        self.assertEqual(latest[0]["story"], "higher id")


class MigrationScriptTests(unittest.TestCase):
    def setUp(self):
        self.sql = MIGRATION.read_text()

    def test_script_does_not_delete_or_rewrite_existing_columns(self):
        stripped = _strip_sql_comments(self.sql)
        self.assertIsNone(re.search(r"\bDELETE\b", stripped, flags=re.I))
        self.assertIsNone(re.search(r"\bTRUNCATE\b", stripped, flags=re.I))
        self.assertIsNone(re.search(r"\bDROP\s+TABLE\b", stripped, flags=re.I))
        self.assertIsNone(re.search(r"\bDROP\s+DATABASE\b", stripped, flags=re.I))
        self.assertNotIn("DROP COLUMN", stripped.upper())

        updates = re.findall(r"\bUPDATE\b.+?;", stripped, flags=re.I | re.S)
        self.assertEqual(len(updates), 1)
        update = updates[0]
        self.assertIn("SET si.attempt_number = ranked.attempt_rank", update)
        self.assertIn("WHERE si.attempt_number IS NULL", update)
        self.assertNotIn("story", update.lower())
        self.assertNotIn("submitted_at", update.lower())

    def test_script_widens_uniqueness_with_attempt_number(self):
        self.assertIn(
            "UNIQUE (student_id, week_id, attempt_number)",
            self.sql,
        )
        self.assertIn("uq_student_week_attempt", self.sql)
        self.assertIn("INDEX_NAME <> 'PRIMARY'", self.sql)
        self.assertIn("'student_id,week_id'", self.sql)
        self.assertIn("student_inputs_latest", self.sql)
        self.assertIn("ALGORITHM=INSTANT", self.sql)


def _strip_sql_comments(sql: str) -> str:
    without_block = re.sub(r"/\*.*?\*/", " ", sql, flags=re.S)
    lines = []
    for line in without_block.splitlines():
        if "--" in line:
            line = line[: line.index("--")]
        lines.append(line)
    return "\n".join(lines)


if __name__ == "__main__":
    unittest.main()
