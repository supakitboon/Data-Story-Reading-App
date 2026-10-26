"""Single-flight guard for the Reflection "Submit Feedback & Send Email" button.

Streamlit delivers each click as its own script run. With ``runner.fastReruns``
(the default) a second click while the first run is inside the database save
or the SMTP send stops that run at its next ``st.*`` or ``session_state``
access and starts another run immediately. Button triggers are also preserved
when those runs are coalesced, so the button is True again on the follow-up
run.

A flag stored only in ``session_state`` cannot close that window: the
interrupted run is not allowed to read or write ``session_state`` after the
stop is requested, so it may finish the email and then die before recording
that it did. ``SubmitFlight`` is ordinary Python state. The run that owns the
save marks the flight done there, and the overlapping run sees that mark and
does not save again.

Phases:

* ``idle`` — a click may start a save.
* ``busy`` — the button is disabled and a spinner is shown while the save runs.
* ``settle`` — show the result and re-enable the button. A click delivered with
  this run is the queued double-click and is ignored. The result is kept.
* ``done`` — keep showing that same result on later script runs, with the
  button enabled, until Restart or another submit.

``on_submit_click`` is the button ``on_click``. Streamlit runs that before the
script body, and it builds a new ``__main__`` module for every run. The flight
therefore has to live in this imported module. A dict on the app script is a
different object by the time the result is drawn, so the email line would be
gone. A click during ``done`` (or ``idle``) starts another attempt and clears
the previous confirmation. The confirmation stays on screen after the spinner
finishes: later script runs draw it again until Restart or another submit.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Mapping, MutableMapping

PHASE_KEY = "feedback_submit_phase"
NOTICES_KEY = "feedback_submit_notices"

IDLE = "idle"
BUSY = "busy"
SETTLE = "settle"
DONE = "done"

BUTTON_KEY = "submit_feedback_email"
RESTART_KEY = "reflection_restart"
SUBMIT_LABEL = "Submit Feedback & Send Email"
SAVING_LABEL = "Saving feedback and sending email…"
SPINNER_TEXT = "Saving your feedback and sending the email…"
SAVE_FAILED_MESSAGE = "❌ Could not save submission. Email not sent."

_notice_target = threading.local()

# Survives the new __main__ module Streamlit creates on every script run.
# A dict defined in the app script does not: the click callback and the
# following run would not share the notices.
_SUBMIT_FLIGHTS: dict[str, SubmitFlight] = {}
_SUBMIT_FLIGHTS_GUARD = threading.Lock()


def get_submit_flight(session_id: str) -> SubmitFlight:
    """Return the save record for one Streamlit session.

    ``session_id`` comes from the script-run context. The same id is used by
    the button callback and by the script body, including the run after the
    save reruns.
    """
    key = session_id or "default"
    with _SUBMIT_FLIGHTS_GUARD:
        flight = _SUBMIT_FLIGHTS.get(key)
        if flight is None:
            flight = SubmitFlight()
            _SUBMIT_FLIGHTS[key] = flight
        return flight


def clear_submit_flights() -> None:
    """Drop every cached flight. Tests use this to isolate sessions."""
    with _SUBMIT_FLIGHTS_GUARD:
        _SUBMIT_FLIGHTS.clear()


class SubmitFlight:
    """Per-session save record that does not go through ``st.session_state``.

    ``lock`` only covers the idle-to-busy arm. ``save_lock`` covers the
    database save and email. They stay separate so the button callback,
    which already holds Streamlit's session lock, never waits on the save.
    """

    def __init__(self) -> None:
        # Phase arm only. Never hold this across a session_state access:
        # the button callback already owns Streamlit's session lock.
        self.lock = threading.Lock()
        # Save/email only. The button callback must not acquire this.
        self.save_lock = threading.Lock()
        self.token = 0
        self.done_token: int | None = None
        self.notices: list[dict[str, Any]] = []
        self.phase = IDLE

    @property
    def save_finished(self) -> bool:
        return self.done_token is not None and self.done_token == self.token


def on_submit_click(
    session_state: MutableMapping[str, Any],
    flight: SubmitFlight,
    lock: Any = None,
) -> None:
    """Arm a save from ``idle`` or ``done``.

    Clicks during ``busy`` or ``settle`` do nothing, so a queued double-click
    cannot start a second save. A click once the result is on screen (``done``)
    clears that confirmation and starts the next attempt.
    """

    def _mutate() -> bool:
        if flight.phase not in (IDLE, DONE):
            return False
        flight.token += 1
        flight.done_token = None
        flight.notices = []
        flight.phase = BUSY
        return True

    if lock is None:
        armed = _mutate()
    else:
        with lock:
            armed = _mutate()
    if armed:
        session_state[PHASE_KEY] = BUSY
        session_state[NOTICES_KEY] = []


def push_notice(
    session_state: MutableMapping[str, Any] | None,
    level: str,
    message: str,
    exc: BaseException | None = None,
) -> bool:
    """Store a status message for the settle run, if a submit is being captured.

    Returns True when the message was stored. Returns False when the caller
    should draw it immediately. The thread-local target is checked first so a
    save can record errors without touching ``session_state`` (an interrupted
    Streamlit run is not allowed to).
    """
    notice = {"level": level, "message": message, "exc": exc}
    target = getattr(_notice_target, "notices", None)
    if target is not None:
        target.append(notice)
        return True
    return False


@contextmanager
def capturing_notices(notices: list[dict[str, Any]]) -> Iterator[None]:
    """Send ``push_notice`` into ``notices`` without touching ``session_state``."""
    previous = getattr(_notice_target, "notices", None)
    _notice_target.notices = notices
    try:
        yield
    finally:
        _notice_target.notices = previous


def show_notice(st: Any, notice: Mapping[str, Any]) -> None:
    level = notice.get("level")
    message = notice.get("message") or ""
    exc = notice.get("exc")
    if level == "success":
        st.success(message)
    elif level == "info":
        st.info(message)
    elif level == "warning":
        st.warning(message)
    elif level == "error":
        st.error(message)
    elif level == "caption":
        st.caption(message)
    elif level == "exception":
        st.exception(exc if exc is not None else Exception(message or "Error"))
    else:
        st.error(message)


def show_notices(st: Any, notices: list[Mapping[str, Any]] | None) -> None:
    for notice in notices or []:
        show_notice(st, notice)


def _rerun(st: Any) -> None:
    """Rerun the whole app so form fields are read again on the next pass.

    ``st.rerun`` signals success with a ``BaseException``.
    """
    st.rerun()


def _mirror_settle(st: Any, flight: SubmitFlight) -> None:
    st.session_state[PHASE_KEY] = SETTLE
    st.session_state[NOTICES_KEY] = list(flight.notices)


def _visible_notices(st: Any, flight: SubmitFlight) -> list[dict[str, Any]]:
    """Notices to draw after the save.

    The flight is the copy made during the save. ``session_state`` is the
    copy that survives when this run's flight object is new and still empty,
    which is what happens if the registry was left on the app script.
    """
    if flight.notices:
        return list(flight.notices)
    stored = st.session_state.get(NOTICES_KEY) or []
    return [dict(notice) for notice in stored]


def run_submit_phase(
    st: Any,
    perform: Callable[[], None],
    flight: SubmitFlight,
    *,
    on_click: Callable[[], None],
    on_restart: Callable[[], None] | None = None,
    lock: Any = None,
) -> None:
    """Draw the submit button and run ``perform`` at most once per armed click.

    ``lock`` serializes overlapping runs from a second click. Hold it only
    around ``perform``; ``perform`` must not call back into this function.
    """
    phase = st.session_state.get(PHASE_KEY, IDLE)
    # The run that did the save may have been stopped before it could copy
    # "settle" into session_state. The flight is the source of truth.
    if flight.phase in (SETTLE, DONE) and phase == BUSY:
        phase = flight.phase
        st.session_state[PHASE_KEY] = phase
        st.session_state[NOTICES_KEY] = list(flight.notices)

    busy = phase == BUSY
    clicked = st.button(
        SAVING_LABEL if busy else SUBMIT_LABEL,
        key=BUTTON_KEY,
        disabled=busy,
        on_click=on_click,
    )

    # If on_click did not run before the body, arm on this pass and rerun so
    # the disabled button is drawn before the save. ``done`` is included so a
    # later intentional click still starts a new attempt.
    if phase in (IDLE, DONE) and clicked:
        on_click()
        _rerun(st)
        return

    if phase == BUSY:
        with st.spinner(SPINNER_TEXT):
            if lock is not None:
                lock.acquire()
            try:
                if not flight.save_finished:
                    try:
                        with capturing_notices(flight.notices):
                            perform()
                    except Exception as exc:
                        flight.notices.append(
                            {"level": "error", "message": SAVE_FAILED_MESSAGE, "exc": None}
                        )
                        flight.notices.append(
                            {"level": "exception", "message": "", "exc": exc}
                        )
                    flight.done_token = flight.token
                flight.phase = SETTLE
            finally:
                if lock is not None:
                    lock.release()
        _mirror_settle(st, flight)
        _rerun(st)
        return

    if phase in (SETTLE, DONE):
        # Draw again on every later script run. Replacing this output is what
        # Streamlit does on the next run, so the notice has to still be here.
        notices = _visible_notices(st, flight)
        show_notices(st, notices)
        if on_restart is not None and st.button("Restart", key=RESTART_KEY):
            flight.phase = IDLE
            flight.notices = []
            st.session_state[PHASE_KEY] = IDLE
            st.session_state[NOTICES_KEY] = []
            on_restart()
            return
        if notices and not flight.notices:
            flight.notices = list(notices)
        if phase == SETTLE:
            flight.phase = DONE
            st.session_state[PHASE_KEY] = DONE
        st.session_state[NOTICES_KEY] = list(notices)
        return
