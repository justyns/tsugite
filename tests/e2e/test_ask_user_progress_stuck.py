"""A live progress bubble must not get orphaned (stuck on the "Working..." spinner)
when an `info` event (from `send_message()`) lands mid-turn before `ask_user`.

Repro for the "code expander stuck showing only working..." report: the streaming
`info` handler pushed a standalone info bubble *after* the live progress without
finalizing it first, so the progress was no longer the last array element. Neither
the `ask_user` handler nor the end-of-stream finalize (both keyed on the *last*
element) could then convert it to `progress-done`, leaving it spinning forever.

`history.js` already finalizes-then-pushes for info; this pins the live path to match.
"""

from .helpers import CONV_VIEW


def _replay(page, events):
    """Drive a realistic ask_user streaming sequence through the real streaming.js
    handlers, mutating the *selected session's* message array so it renders in the DOM.

    Pseudo-events 'ask_user' / 'submit_last' / 'done' mirror what sendMessage()'s SSE
    loop does inline; everything else goes through _handleProgressEvent like a real
    progress event.
    """
    return page.evaluate(
        """([sel, events]) => {
            const v = Alpine.$data(document.querySelector(sel));
            const sid = v.selectedSessionId;
            const arr = () => v._sessionState(sid).messages;
            for (const ev of events) {
                if (ev.type === 'ask_user') {
                    v._finalizeLiveProgress(arr());
                    arr().push({ type: 'ask_user', question: ev.question, questionType: 'text',
                                 options: [], answered: false, answer: '', inputValue: '' });
                } else if (ev.type === 'submit_last') {
                    const a = arr();
                    for (let i = a.length - 1; i >= 0; i--) {
                        if (a[i].type === 'ask_user' && !a[i].answered) { a[i].answered = true; a[i].answer = ev.answer; break; }
                    }
                } else if (ev.type === 'done') {
                    // Mirrors the end-of-stream finalize in sendMessage().
                    v._finalizeLiveProgress(arr());
                } else {
                    v._handleProgressEvent(ev, sid);
                }
            }
            return arr().map(m => ({ type: m.type, statusText: m.statusText || '' }));
        }""",
        [CONV_VIEW, events],
    )


def test_info_then_ask_user_does_not_strand_live_progress(chat_page):
    page = chat_page

    # An agent turn whose code block calls send_message(...) (emits `info`) and then
    # ask_user(...), exactly as exercised during ask-user tool testing.
    sequence = [
        {"type": "turn_start", "turn": 1},
        {"type": "code", "content": 'send_message("Checking..."); answer = ask_user(question="Proceed?")'},
        {"type": "info", "message": "Checking..."},
        {"type": "ask_user", "question": "Proceed?"},
        {"type": "submit_last", "answer": "yes"},
        {"type": "tool_result", "tool": "unknown", "output": "answer = 'yes'", "success": True},
        {"type": "done"},
    ]
    bubbles = _replay(page, sequence)

    # Capture the rendered thread regardless of pass/fail (repro vs fix screenshot).
    page.screenshot(path="/tmp/tsugite-issue-state.png", full_page=True)

    # Once the stream ends, every code expander must reflect a completed state:
    stuck = [b for b in bubbles if b["type"] == "progress"]
    assert not stuck, (
        f"{len(stuck)} progress bubble(s) left live (stuck on the spinner) after the stream "
        "ended; the info event orphaned the live progress"
    )
    # ...and none may still advertise the in-flight "Working..." status next to its ✓.
    working = [b for b in bubbles if "Working" in b["statusText"]]
    assert not working, f"finalized code expander still shows a 'Working...' status: {working}"


def test_info_mid_turn_finalizes_progress_without_ask_user(chat_page):
    """The same orphan also bites a plain send_message() (no ask_user in the turn)."""
    page = chat_page
    bubbles = _replay(
        page,
        [
            {"type": "turn_start", "turn": 1},
            {"type": "code", "content": 'send_message("halfway done")'},
            {"type": "info", "message": "halfway done"},
            {"type": "tool_result", "tool": "unknown", "output": "ok", "success": True},
            {"type": "done"},
        ],
    )
    assert not [b for b in bubbles if b["type"] == "progress"], "live progress stranded by info"
    assert not [b for b in bubbles if "Working" in b["statusText"]], "stale 'Working...' on a done bubble"


def test_info_with_no_live_progress_makes_no_phantom_progress(chat_page):
    """An info event with nothing in flight should be a bare info bubble, not a
    phantom (empty) progress bubble. The pre-fix handler ran _ensureLiveProgress
    first, which would have conjured one."""
    page = chat_page
    bubbles = _replay(page, [{"type": "info", "message": "just an fyi"}])
    assert [b["type"] for b in bubbles] == ["info"], f"expected only an info bubble, got {bubbles}"
