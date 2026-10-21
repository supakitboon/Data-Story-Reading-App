-- Keep every student submission
-- ==============================
-- Review this whole file, then run it yourself in DBeaver against the
-- APPLICATION database (the one the Streamlit app uses).
--
-- How to run in DBeaver:
--   1. Open a connection to the app database and select that schema.
--   2. Run PART A first (read-only). Read the result grids.
--   3. Run PART B with "Execute SQL Script" (Alt+X), not "Execute statement".
--      PART B uses DELIMITER, which the single-statement runner ignores.
--
-- What this changes
-- -----------------
-- student_inputs only:
--   * adds attempt_number (INT NULL, then backfilled, then NOT NULL)
--   * adds submitted_at (DATETIME NULL, left NULL on existing rows)
--   * backfills attempt_number on existing rows and no other column
--   * adds UNIQUE (student_id, week_id, attempt_number)
--   * drops a UNIQUE index on student_inputs ONLY when its columns are
--     exactly (student_id) or exactly (student_id, week_id) in either order
--   * creates two views: student_attempts and student_inputs_latest
--
-- What this does NOT do
-- ---------------------
--   * no DELETE, no TRUNCATE, no DROP TABLE, no DROP DATABASE
--   * no UPDATE of story, scores, names, email, class, reflection, or ids
--   * no change to student_sentences (old sentence rows stay on their input_id)
--   * no change to students.email UNIQUE or weeks.week_number UNIQUE
--   * no change to the student_inputs primary key (input_id)
--
-- Why existing rows stay intact
-- -----------------------------
-- The only column write is:
--
--     UPDATE student_inputs SET attempt_number = <rank> WHERE attempt_number IS NULL
--
-- Rank is 1 for the earliest input_id of each (student_id, week_id). If the
-- live database really has at most one row per student and week — which is
-- what a unique key on student_id, or on (student_id, week_id), guarantees —
-- every existing row gets attempt_number = 1. Later rows, if any already
-- exist, get 2, 3, ... so the new unique key can be created without deleting
-- them. Forcing every row to 1 would make that unique key fail whenever a
-- duplicate pair was already stored.
--
-- submitted_at is not backfilled. There is no original timestamp column in
-- the schema this repo documents, and stamping NOW() onto old work would
-- invent a submission time.
--
-- MySQL may rebuild student_inputs when attempt_number is changed to NOT NULL.
-- That copies rows through as they are. It does not change any pre-existing
-- column. Adding the nullable columns uses ALGORITHM=INSTANT so that step
-- does not rewrite the table. If INSTANT is not supported, that statement
-- fails and stops the script instead of falling back to a table copy.
--
-- Assumptions to double-check in PART A (this repo has no CREATE TABLE)
-- -------------------------------------------------------------------
-- The application code and README are the only schema sources. Confirm:
--   1. You are connected to the app database, and student_inputs is there.
--   2. input_id is the primary key and is AUTO_INCREMENT.
--   3. ENGINE is InnoDB for student_inputs and student_sentences.
--      The old app deleted a row and then inserted. On InnoDB that pair is
--      one transaction, so a failed insert rolls the delete back. On MyISAM
--      it would not. Do not leave the old app running after this migration
--      if the engine is not InnoDB.
--   4. The unique-index grid. This script only relaxes UNIQUE indexes whose
--      column list is exactly student_id, or exactly student_id + week_id.
--      Any other UNIQUE index is left in place. If a second submit still
--      fails with a duplicate-key error, that grid shows the index to widen
--      the same way (add attempt_number) — do not drop it blindly.
--   5. MySQL 8.0+ or MariaDB 10.2+ (the backfill uses ROW_NUMBER).
--   6. There is no table already named student_attempts or student_inputs_latest.
--
-- Why the new unique key includes week_id
-- ---------------------------------------
-- The app replaces a submission per student AND week, and students submit
-- once each week. Attempt numbers therefore restart every week.
-- UNIQUE (student_id, attempt_number) would reject week 2 after both weeks
-- were backfilled as attempt 1. UNIQUE (student_id, week_id, attempt_number)
-- keeps one row per attempt and still allows every week.
-- If the live unique key is only student_id, widening it to
-- (student_id, week_id, attempt_number) is a relaxation: the single existing
-- row still satisfies it, and a second week or a second attempt becomes legal.
-- The primary key stays input_id, so student_sentences.input_id does not change.
--
-- Deploy together with the app change
-- -----------------------------------
-- This script does not stop the old application from deleting a previous row.
-- The new application inserts a new row and never deletes. Ship both before
-- students submit again.
--   * New app, old schema: the insert names attempt_number and fails closed.
--     Nothing is deleted.
--   * Old app, new schema, InnoDB: the old insert omits attempt_number, fails
--     the NOT NULL constraint, and rolls back the delete.
--   * Old app, new schema, MyISAM: a failed insert may NOT restore the delete.
--     Confirm InnoDB in PART A before running PART B.


-- =============================================================================
-- PART A — read only. Safe to run more than once. Changes nothing.
-- =============================================================================

SELECT DATABASE() AS current_database_double_check_this;

SELECT TABLE_NAME, ENGINE
FROM information_schema.TABLES
WHERE TABLE_SCHEMA = DATABASE()
  AND TABLE_NAME IN ('students', 'weeks', 'student_inputs', 'student_sentences');

SELECT
  INDEX_NAME,
  NON_UNIQUE,
  GROUP_CONCAT(COLUMN_NAME ORDER BY SEQ_IN_INDEX SEPARATOR ', ') AS columns_in_order
FROM information_schema.STATISTICS
WHERE TABLE_SCHEMA = DATABASE()
  AND TABLE_NAME = 'student_inputs'
GROUP BY INDEX_NAME, NON_UNIQUE
ORDER BY INDEX_NAME;

-- Read this before PART B. An empty result means every student+week has one
-- row, so the backfill sets attempt_number = 1 on every existing submission.
-- A row here is still safe: those submissions are numbered 1, 2, 3... by
-- input_id and none of them are deleted. After students submit more than
-- once, this list is the attempts you kept. Re-running PART B does not
-- renumber a row that already has attempt_number set.
SELECT student_id, week_id, COUNT(*) AS rows_for_this_student_week
FROM student_inputs
GROUP BY student_id, week_id
HAVING COUNT(*) > 1;

SELECT COUNT(*) AS existing_submission_rows FROM student_inputs;
SELECT COUNT(*) AS existing_sentence_rows FROM student_sentences;


-- =============================================================================
-- PART B — apply. Run after reading PART A.
-- =============================================================================

DROP PROCEDURE IF EXISTS apply_keep_submission_attempts;

DELIMITER $$

CREATE PROCEDURE apply_keep_submission_attempts()
BEGIN
  DECLARE dup_groups INT DEFAULT 0;
  DECLARE missing_attempts INT DEFAULT 0;
  DECLARE version_major INT DEFAULT 0;

  DECLARE EXIT HANDLER FOR SQLEXCEPTION
  BEGIN
    ROLLBACK;
    RESIGNAL;
  END;

  SET version_major = CAST(SUBSTRING_INDEX(@@version, '.', 1) AS UNSIGNED);
  IF version_major < 8 AND @@version NOT LIKE '%MariaDB%' THEN
    SIGNAL SQLSTATE '45000'
      SET MESSAGE_TEXT = 'MySQL 8.0+ or MariaDB 10.2+ is required (ROW_NUMBER). No table change has been made yet.';
  END IF;

  IF NOT EXISTS (
    SELECT 1
    FROM information_schema.TABLES
    WHERE TABLE_SCHEMA = DATABASE()
      AND TABLE_NAME = 'student_inputs'
  ) THEN
    SIGNAL SQLSTATE '45000'
      SET MESSAGE_TEXT = 'student_inputs is not in the current database. Select the app database and run this again. No change has been made.';
  END IF;

  IF NOT EXISTS (
    SELECT 1
    FROM information_schema.COLUMNS
    WHERE TABLE_SCHEMA = DATABASE()
      AND TABLE_NAME = 'student_inputs'
      AND COLUMN_NAME = 'attempt_number'
  ) THEN
    ALTER TABLE student_inputs
      ADD COLUMN attempt_number INT NULL
        COMMENT 'Attempt within (student_id, week_id). 1 = earliest surviving row. Backfilled; other columns are not modified.',
      ALGORITHM=INSTANT;
  END IF;

  IF NOT EXISTS (
    SELECT 1
    FROM information_schema.COLUMNS
    WHERE TABLE_SCHEMA = DATABASE()
      AND TABLE_NAME = 'student_inputs'
      AND COLUMN_NAME = 'submitted_at'
  ) THEN
    ALTER TABLE student_inputs
      ADD COLUMN submitted_at DATETIME NULL
        COMMENT 'UTC time this attempt was inserted. NULL on rows created before this column existed; not backfilled.',
      ALGORITHM=INSTANT;
  END IF;

  -- Number only rows that do not have an attempt yet. Earliest input_id
  -- in each student+week is 1. Re-running does not renumber rows that
  -- already have an attempt_number.
  DROP TEMPORARY TABLE IF EXISTS tmp_attempt_backfill;
  CREATE TEMPORARY TABLE tmp_attempt_backfill AS
  SELECT
    input_id,
    ROW_NUMBER() OVER (
      PARTITION BY student_id, week_id
      ORDER BY input_id
    ) AS attempt_rank
  FROM student_inputs;

  UPDATE student_inputs AS si
  JOIN tmp_attempt_backfill AS ranked ON ranked.input_id = si.input_id
  SET si.attempt_number = ranked.attempt_rank
  WHERE si.attempt_number IS NULL;

  DROP TEMPORARY TABLE IF EXISTS tmp_attempt_backfill;

  SELECT COUNT(*) INTO missing_attempts
  FROM student_inputs
  WHERE attempt_number IS NULL;

  IF missing_attempts > 0 THEN
    SIGNAL SQLSTATE '45000'
      SET MESSAGE_TEXT = 'attempt_number is still NULL on some rows, so the NOT NULL / unique change was not applied. Existing column values were not modified.';
  END IF;

  SELECT COUNT(*) INTO dup_groups
  FROM (
    SELECT student_id, week_id, attempt_number
    FROM student_inputs
    GROUP BY student_id, week_id, attempt_number
    HAVING COUNT(*) > 1
  ) AS duplicated_attempts;

  IF dup_groups > 0 THEN
    SIGNAL SQLSTATE '45000'
      SET MESSAGE_TEXT = 'Duplicate (student_id, week_id, attempt_number) values exist, so the unique key was not added and no old unique index was dropped.';
  END IF;

  -- The following ALTER commits the backfill (MySQL implicit commit on DDL).
  IF EXISTS (
    SELECT 1
    FROM information_schema.COLUMNS
    WHERE TABLE_SCHEMA = DATABASE()
      AND TABLE_NAME = 'student_inputs'
      AND COLUMN_NAME = 'attempt_number'
      AND IS_NULLABLE = 'YES'
  ) THEN
    ALTER TABLE student_inputs
      MODIFY COLUMN attempt_number INT NOT NULL
        COMMENT 'Attempt within (student_id, week_id). 1 = earliest surviving row.';
  END IF;

  -- Add the widened key BEFORE dropping a narrower one, so a failure here
  -- leaves the old unique index in place.
  IF NOT EXISTS (
    SELECT 1
    FROM information_schema.STATISTICS
    WHERE TABLE_SCHEMA = DATABASE()
      AND TABLE_NAME = 'student_inputs'
      AND INDEX_NAME = 'uq_student_week_attempt'
  ) THEN
    ALTER TABLE student_inputs
      ADD CONSTRAINT uq_student_week_attempt
      UNIQUE (student_id, week_id, attempt_number);
  END IF;

  -- Drop UNIQUE indexes that cannot store a second attempt.
  -- Names are collected first, then dropped. Doing the DROP while a cursor
  -- on information_schema is still open is unreliable.
  -- Allowlist (exact column list, in index order):
  --   student_id
  --   student_id, week_id
  --   week_id, student_id
  -- PRIMARY (input_id) is never selected. Neither is any index that includes
  -- another column, including the new uq_student_week_attempt index.
  SET @indexes_to_drop = NULL;
  SELECT GROUP_CONCAT(INDEX_NAME ORDER BY INDEX_NAME SEPARATOR ',')
    INTO @indexes_to_drop
  FROM (
    SELECT INDEX_NAME
    FROM (
      SELECT
        INDEX_NAME,
        GROUP_CONCAT(COLUMN_NAME ORDER BY SEQ_IN_INDEX SEPARATOR ',') AS cols,
        MAX(NON_UNIQUE) AS non_unique
      FROM information_schema.STATISTICS
      WHERE TABLE_SCHEMA = DATABASE()
        AND TABLE_NAME = 'student_inputs'
        AND INDEX_NAME <> 'PRIMARY'
      GROUP BY INDEX_NAME
    ) AS indexes_on_inputs
    WHERE non_unique = 0
      AND cols IN ('student_id', 'student_id,week_id', 'week_id,student_id')
  ) AS to_drop;

  WHILE @indexes_to_drop IS NOT NULL AND @indexes_to_drop <> '' DO
    SET @idx_to_drop = SUBSTRING_INDEX(@indexes_to_drop, ',', 1);
    IF CHAR_LENGTH(@indexes_to_drop) = CHAR_LENGTH(@idx_to_drop) THEN
      SET @indexes_to_drop = NULL;
    ELSE
      SET @indexes_to_drop = SUBSTRING(
        @indexes_to_drop,
        CHAR_LENGTH(@idx_to_drop) + 2
      );
    END IF;

    SET @drop_sql = CONCAT(
      'ALTER TABLE student_inputs DROP INDEX `',
      REPLACE(@idx_to_drop, '`', '``'),
      '`'
    );
    PREPARE drop_stmt FROM @drop_sql;
    EXECUTE drop_stmt;
    DEALLOCATE PREPARE drop_stmt;
  END WHILE;
END$$

DELIMITER ;

CALL apply_keep_submission_attempts();

DROP PROCEDURE IF EXISTS apply_keep_submission_attempts;

-- Latest attempt per student per week, plus every attempt with its number.
-- is_latest = 1 is the row a one-result-per-student report should use.
-- Attempt order is attempt_number, not submitted_at: older rows have no
-- timestamp, and NULL timestamps must not sort as the newest work.
CREATE OR REPLACE VIEW student_attempts AS
SELECT
  si.*,
  CASE
    WHEN si.input_id = (
      SELECT s2.input_id
      FROM student_inputs AS s2
      WHERE s2.student_id = si.student_id
        AND s2.week_id = si.week_id
      ORDER BY (s2.attempt_number IS NULL) ASC,
               s2.attempt_number DESC,
               s2.input_id DESC
      LIMIT 1
    ) THEN 1
    ELSE 0
  END AS is_latest
FROM student_inputs AS si;

CREATE OR REPLACE VIEW student_inputs_latest AS
SELECT *
FROM student_attempts
WHERE is_latest = 1;

-- =============================================================================
-- PART C — read only, after PART B. Confirm the backfill and the new key.
-- =============================================================================

SELECT
  SUM(attempt_number = 1) AS rows_marked_attempt_1,
  SUM(attempt_number <> 1) AS rows_with_a_later_attempt_number,
  COUNT(*) AS total_rows
FROM student_inputs;

SELECT
  INDEX_NAME,
  NON_UNIQUE,
  GROUP_CONCAT(COLUMN_NAME ORDER BY SEQ_IN_INDEX SEPARATOR ', ') AS columns_in_order
FROM information_schema.STATISTICS
WHERE TABLE_SCHEMA = DATABASE()
  AND TABLE_NAME = 'student_inputs'
GROUP BY INDEX_NAME, NON_UNIQUE
ORDER BY INDEX_NAME;

-- One row per student per week (the previous "current result"):
--   SELECT * FROM student_inputs_latest;
-- Every attempt, oldest first, with sentences:
--   SELECT a.student_id, a.week_id, a.attempt_number, a.submitted_at, a.is_latest, s.*
--   FROM student_attempts AS a
--   JOIN student_sentences AS s ON s.input_id = a.input_id
--   ORDER BY a.student_id, a.week_id, a.attempt_number, s.sentence_idx;
