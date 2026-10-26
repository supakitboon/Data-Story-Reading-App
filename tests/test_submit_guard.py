"""The Reflection submit button saves and emails once per intentional click."""

from __future__ import annotations

import threading
import time
import unittest
from contextlib import contextmanager
from pathlib import Path

from utils.submit_guard import (
    BUTTON_KEY,
    DONE,
    IDLE,
    NOTICES_KEY,
    PHASE_KEY,
    RESTART_KEY,
    SAVING_LABEL,
    SETTLE,
    SPINNER_TEXT,
    SUBMIT_LABEL,
    SubmitFlight,
    clear_submit_flights,
    get_submit_flight,
    on_submit_click,
    push_notice,
    run_submit_phase,
)

EMAIL_SENT = "✅ Email sent to ab…@school.edu"


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "streamlit_predict_app.py"
REMOVED_SUCCESS = (
    "and the email was sent. Earlier submissions for this week are kept."
)


class _Rerun(BaseException):
    """Stands in for Streamlit's rerun signal, which is not an Exception."""


class FakeSt:
    def __init__(self):
        self.session_state = {}
        self.events = []
        self.click_return = False
        self.restart_return = False
        self._events_lock = threading.Lock()

    def _record(self, event):
        with self._events_lock:
            self.events.append(event)

    def button(self, label, key=None, disabled=False, on_click=None):
        self._record(("button", label, disabled, key))
        if key == RESTART_KEY:
            return self.restart_return
        return self.click_return

    def spinner(self, text):
        self._record(("spinner", text))

        @contextmanager
        def _cm():
            yield

        return _cm()

    def rerun(self, scope="app"):
        self._record(("rerun", scope))
        raise _Rerun()

    def success(self, message):
        self._record(("success", message))

    def info(self, message):
        self._record(("info", message))

    def warning(self, message):
        self._record(("warning", message))

    def error(self, message):
        self._record(("error", message))

    def caption(self, message):
        self._record(("caption", message))

    def exception(self, exc):
        self._record(("exception", exc))


def _drive(st, flight, perform, *, prerun_click=True, on_restart=None):
    def on_click():
        on_submit_click(st.session_state, flight, flight.lock)

    if prerun_click:
        on_click()
    try:
        run_submit_phase(
            st,
            perform,
            flight,
            on_click=on_click,
            on_restart=on_restart,
            lock=flight.save_lock,
        )
    except _Rerun:
        return "rerun"
    return "done"


class SubmitOnceTests(unittest.TestCase):
    def test_one_click_saves_once_and_reenabled_button_shows_email_notice(self):
        st = FakeSt()
        flight = SubmitFlight()
        saves = []

        def perform():
            saves.append("save")
            push_notice(None, "success", EMAIL_SENT)

        self.assertEqual(_drive(st, flight, perform), "rerun")
        self.assertEqual(saves, ["save"])
        self.assertEqual(flight.phase, "settle")
        self.assertIn(
            ("button", SAVING_LABEL, True, BUTTON_KEY),
            st.events,
        )
        self.assertIn(("spinner", SPINNER_TEXT), st.events)
        self.assertLess(
            st.events.index(("button", SAVING_LABEL, True, BUTTON_KEY)),
            st.events.index(("spinner", SPINNER_TEXT)),
        )

        st.events.clear()
        self.assertEqual(
            _drive(st, flight, perform, prerun_click=False, on_restart=lambda: None),
            "done",
        )
        self.assertEqual(saves, ["save"])
        self.assertEqual(flight.phase, DONE)
        self.assertEqual(st.session_state[PHASE_KEY], DONE)
        self.assertEqual(flight.notices[0]["message"], EMAIL_SENT)
        self.assertEqual(st.session_state[NOTICES_KEY][0]["message"], EMAIL_SENT)
        self.assertIn(("button", SUBMIT_LABEL, False, BUTTON_KEY), st.events)
        self.assertIn(("success", EMAIL_SENT), st.events)
        self.assertIn(("button", "Restart", False, RESTART_KEY), st.events)
        self.assertFalse(any(REMOVED_SUCCESS in str(event) for event in st.events))

    def test_email_sent_notice_stays_visible_on_the_following_run(self):
        st = FakeSt()
        flight = SubmitFlight()
        saves = []

        def perform():
            saves.append("save")
            push_notice(None, "success", EMAIL_SENT)

        self.assertEqual(_drive(st, flight, perform), "rerun")
        self.assertEqual(
            _drive(st, flight, perform, prerun_click=False, on_restart=lambda: None),
            "done",
        )
        self.assertEqual(flight.phase, DONE)
        self.assertTrue(flight.notices)

        st.events.clear()
        self.assertEqual(
            _drive(st, flight, perform, prerun_click=False, on_restart=lambda: None),
            "done",
        )
        self.assertEqual(saves, ["save"])
        self.assertEqual(flight.phase, DONE)
        self.assertEqual(flight.notices[0]["message"], EMAIL_SENT)
        self.assertEqual(st.session_state[NOTICES_KEY][0]["message"], EMAIL_SENT)
        self.assertIn(("success", EMAIL_SENT), st.events)
        self.assertIn(("button", SUBMIT_LABEL, False, BUTTON_KEY), st.events)
        self.assertIn(("button", "Restart", False, RESTART_KEY), st.events)
        self.assertFalse(any(REMOVED_SUCCESS in str(event) for event in st.events))

    def test_second_click_while_busy_does_not_save_again(self):
        st = FakeSt()
        flight = SubmitFlight()
        saves = []

        def perform():
            saves.append("save")
            push_notice(None, "success", EMAIL_SENT)

        def on_click():
            on_submit_click(st.session_state, flight, flight.lock)

        on_click()
        on_click()
        self.assertEqual(flight.token, 1)

        self.assertEqual(_drive(st, flight, perform, prerun_click=False), "rerun")
        self.assertEqual(saves, ["save"])

        # Queued extra click arrives with the settle run.
        on_click()
        self.assertEqual(flight.token, 1)
        self.assertEqual(_drive(st, flight, perform, prerun_click=False), "done")
        self.assertEqual(saves, ["save"])
        self.assertEqual(flight.phase, DONE)
        self.assertEqual(flight.notices[0]["message"], EMAIL_SENT)

    def test_later_click_saves_another_attempt(self):
        st = FakeSt()
        flight = SubmitFlight()
        saves = []

        def perform():
            saves.append("save")

        _drive(st, flight, perform)
        _drive(st, flight, perform, prerun_click=False)
        _drive(st, flight, perform)
        self.assertEqual(saves, ["save", "save"])

    def test_email_failure_keeps_the_warning_and_not_the_success_banner(self):
        st = FakeSt()
        flight = SubmitFlight()
        warning = (
            "Feedback was saved as attempt 1 "
            "for Week 2, but the email could not be sent. "
            "Earlier submissions for this week are kept."
        )

        def perform():
            push_notice(None, "error", "❌ Failed to send email.")
            push_notice(None, "warning", warning)

        _drive(st, flight, perform)
        _drive(st, flight, perform, prerun_click=False)
        st.events.clear()
        _drive(st, flight, perform, prerun_click=False)
        self.assertEqual(flight.phase, DONE)
        self.assertIn(("warning", warning), st.events)
        self.assertIn(("error", "❌ Failed to send email."), st.events)
        self.assertFalse(any(event[0] == "success" for event in st.events))
        self.assertFalse(any(REMOVED_SUCCESS in str(event) for event in st.events))

    def test_overlapping_runs_save_once(self):
        st = FakeSt()
        flight = SubmitFlight()
        saves = []

        def perform():
            time.sleep(0.05)
            saves.append("save")
            push_notice(None, "success", "✅ Email sent to ab…@school.edu")

        def on_click():
            on_submit_click(st.session_state, flight, flight.lock)

        on_click()
        on_click()

        errors = []

        def worker():
            try:
                run_submit_phase(
                    st,
                    perform,
                    flight,
                    on_click=on_click,
                    lock=flight.save_lock,
                )
            except _Rerun:
                pass
            except Exception as exc:  # pragma: no cover - failure path
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=3)
            self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(saves, ["save"])
        self.assertTrue(flight.save_finished)

    def test_click_without_callback_arms_and_does_not_save_yet(self):
        st = FakeSt()
        st.click_return = True
        flight = SubmitFlight()
        saves = []

        self.assertEqual(
            _drive(st, flight, lambda: saves.append("save"), prerun_click=False),
            "rerun",
        )
        self.assertEqual(saves, [])
        self.assertEqual(flight.phase, "busy")

        st.click_return = False
        self.assertEqual(
            _drive(st, flight, lambda: saves.append("save"), prerun_click=False),
            "rerun",
        )
        self.assertEqual(saves, ["save"])

    def test_perform_error_is_settled_once(self):
        st = FakeSt()
        flight = SubmitFlight()
        calls = []

        def perform():
            calls.append("save")
            raise RuntimeError("db down")

        _drive(st, flight, perform)
        self.assertEqual(calls, ["save"])
        self.assertTrue(flight.save_finished)
        _drive(st, flight, perform, prerun_click=False)
        st.events.clear()
        _drive(st, flight, perform, prerun_click=False)
        self.assertEqual(calls, ["save"])
        self.assertEqual(flight.phase, DONE)
        self.assertIn("db down", str(st.events))

    def test_app_drops_the_attempt_success_banner_and_keeps_failure_copy(self):
        source = APP.read_text()
        self.assertNotIn(REMOVED_SUCCESS, source)
        self.assertIn("but the email could not be sent.", source)
        self.assertIn("Earlier submissions for this week are kept.", source)
        self.assertIn("Email sent to", source)
        self.assertIn("Could not resolve student/week. Aborting save.", source)
        self.assertIn("❌ Could not save submission. Email not sent.", source)
        self.assertIn("get_submit_flight", source)
        self.assertNotIn("_SUBMIT_FLIGHTS", source)

    def test_fresh_flight_still_shows_the_mirrored_email_notice(self):
        # Streamlit execs the app as a new __main__ each run. A flight dict
        # on that script is empty on the settle run; the session copy is what
        # the busy run managed to store.
        st = FakeSt()
        saves = []

        def perform():
            saves.append("save")
            push_notice(None, "success", EMAIL_SENT)

        busy = SubmitFlight()
        self.assertEqual(_drive(st, busy, perform), "rerun")
        self.assertEqual(st.session_state[PHASE_KEY], SETTLE)
        self.assertEqual(st.session_state[NOTICES_KEY][0]["message"], EMAIL_SENT)

        settled = SubmitFlight()
        st.events.clear()
        self.assertEqual(
            _drive(st, settled, perform, prerun_click=False, on_restart=lambda: None),
            "done",
        )
        self.assertEqual(saves, ["save"])
        self.assertIn(("success", EMAIL_SENT), st.events)
        self.assertEqual(st.session_state[PHASE_KEY], DONE)
        self.assertEqual(st.session_state[NOTICES_KEY][0]["message"], EMAIL_SENT)

        later = SubmitFlight()
        st.events.clear()
        self.assertEqual(
            _drive(st, later, perform, prerun_click=False, on_restart=lambda: None),
            "done",
        )
        self.assertEqual(saves, ["save"])
        self.assertEqual(later.phase, IDLE)
        self.assertIn(("success", EMAIL_SENT), st.events)
        self.assertEqual(st.session_state[NOTICES_KEY][0]["message"], EMAIL_SENT)
        self.assertFalse(any(REMOVED_SUCCESS in str(event) for event in st.events))

    def test_session_flight_keeps_the_notice_across_lookups(self):
        clear_submit_flights()
        self.addCleanup(clear_submit_flights)
        st = FakeSt()
        saves = []

        def perform():
            saves.append("save")
            push_notice(None, "success", EMAIL_SENT)

        flight = get_submit_flight("student-session")
        self.assertEqual(_drive(st, flight, perform), "rerun")
        again = get_submit_flight("student-session")
        self.assertIs(again, flight)
        st.events.clear()
        self.assertEqual(
            _drive(st, again, perform, prerun_click=False, on_restart=lambda: None),
            "done",
        )
        third = get_submit_flight("student-session")
        st.events.clear()
        self.assertEqual(
            _drive(st, third, perform, prerun_click=False, on_restart=lambda: None),
            "done",
        )
        self.assertEqual(saves, ["save"])
        self.assertEqual(third.phase, DONE)
        self.assertEqual(third.notices[0]["message"], EMAIL_SENT)
        self.assertIn(("success", EMAIL_SENT), st.events)

    def test_restart_clears_the_email_sent_notice(self):
        st = FakeSt()
        flight = SubmitFlight()

        def perform():
            push_notice(None, "success", EMAIL_SENT)

        _drive(st, flight, perform)
        _drive(st, flight, perform, prerun_click=False, on_restart=lambda: None)
        self.assertEqual(flight.phase, DONE)

        st.restart_return = True
        restarted = []
        _drive(
            st,
            flight,
            perform,
            prerun_click=False,
            on_restart=lambda: restarted.append("restart"),
        )
        self.assertEqual(restarted, ["restart"])
        self.assertEqual(flight.phase, IDLE)
        self.assertEqual(flight.notices, [])
        self.assertEqual(st.session_state[PHASE_KEY], IDLE)
        self.assertEqual(st.session_state[NOTICES_KEY], [])


if __name__ == "__main__":
    unittest.main()
