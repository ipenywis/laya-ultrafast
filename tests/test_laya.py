"""Offline contracts for the local Laya policy. A fake stands in for the model; nothing is downloaded."""

import datetime
import time
from unittest.mock import Mock

import pytest

from laya_ultrafast import agent as loop
from laya_ultrafast import laya, model
from laya_ultrafast.browser import fingerprint


class FakeLaya:
    """Answers each choice question with `prefer(question_id, criteria)`, or the first option."""

    def __init__(self, prefer=None):
        self.prefer = prefer or (lambda _qid, criteria: next(iter(criteria)))
        self.calls = []

    def system_one(self, state, questions):
        self.calls.append((state, questions))
        answers = {}
        for qid, q in questions.items():
            labels = list(q["criteria"])
            choice = self.prefer(qid, q["criteria"])
            rest = (1 - 0.7) / max(1, len(labels) - 1)
            probabilities = {label: 0.7 if label == choice else rest for label in labels}
            if len(labels) == 1:
                probabilities = {choice: 1.0}
            answers[qid] = {"choice": choice, "probabilities": probabilities, "confidence": 0.5}
        return {"answers": answers, "usage": {"input_tokens": 10}}


def page(actions, url="https://example.test/", title="Search", text="Search"):
    state = {"url": url, "title": title, "text": text, "scroll": {"y": 0}, "actions": actions}
    state["fingerprint"] = fingerprint(state)
    return state


FORM = [
    {"id": "e1", "kind": "fill", "label": "Destination", "role": "searchbox", "value": "", "node": 1},
    {"id": "e2", "kind": "click", "label": "Open Destination", "role": "searchbox", "value": "", "node": 1},
    {"id": "e3", "kind": "click", "label": "Find stays", "role": "button", "value": "", "node": 2},
    {"id": "e4", "kind": "click", "label": "View Casa Flora", "role": "button", "value": "", "node": 3},
    {"id": "wait", "kind": "wait", "label": "Wait for the page to update"},
]
PLAN = {"requirements": [{"what": "destination", "value": "Lisbon"}], "open": "Casa Flora", "finish": "It is open."}


@pytest.fixture
def fake(monkeypatch):
    f = FakeLaya()
    monkeypatch.setattr(laya, "laya", lambda: f)
    return f


def policy(plan=PLAN):
    p = laya.LayaPolicy("Find a stay in Lisbon and open Casa Flora.")
    p.plan, p.plan_meta = plan, {"model": "test", "latency_ms": 1}
    return p


def executed(history, decision, label=""):
    history.append({"choice": decision["choice"], "action": label})


def test_a_requirement_named_by_its_field_label_needs_no_mapping_call(fake):
    p = policy({"requirements": [{"what": "Destination", "value": "Lisbon"}], "open": None, "finish": "Seen."})
    d = p.choose(page(FORM), [])
    assert d["choice"] == "e1" and not any(k.startswith("field_") for _s, q in fake.calls for k in q)


def test_goal_values_are_typed_without_a_per_field_text_call(fake):
    p = policy()
    d = p.choose(page(FORM), [])
    assert (d["operation"], d["choice"], d["text"]) == ("TYPE_TEXT", "e1", "Lisbon")
    assert d["target"] == "1"


def test_typed_text_is_submitted_before_opening_a_result(fake):
    p, history = policy(), []
    executed(history, p.choose(page(FORM), history))
    filled = [dict(a, value="Lisbon") if a.get("node") == 1 else a for a in FORM]
    d = p.choose(page(filled), history)
    assert (d["operation"], d["choice"]) == ("CLICK", "e3")  # Find stays, not View Casa Flora


def test_item_page_title_finishes_the_goal(fake):
    p = policy({"requirements": [], "open": "Casa Flora", "finish": "Casa Flora is open."})
    d = p.choose(page(FORM, title="Casa Flora · Forma"), [])
    assert d["operation"] == "DONE" and d["choice"] == "DONE"


def test_a_search_results_title_is_not_the_item_page():
    assert laya.titled("Casa Flora · Forma", "Casa Flora")
    assert not laya.titled("Casa Flora - Search results - Forma", "Casa Flora")


def test_every_target_is_an_observed_action(fake):
    fake.prefer = lambda qid, criteria: list(criteria)[-1]
    p = policy({"requirements": [], "open": None, "finish": "Done."})
    d = p.choose(page(FORM), [])
    assert d["choice"] in {a["id"] for a in FORM} | {"DONE", "BLOCKED"}


def test_invalid_model_answer_executes_nothing(monkeypatch):
    broken = Mock(system_one=Mock(return_value={
        "answers": {"field_0": {"choice": "999", "probabilities": {"999": 1.0}, "confidence": 1.0}},
        "usage": {"input_tokens": 1},
    }))
    monkeypatch.setattr(laya, "laya", lambda: broken)
    form = [dict(a) for a in FORM] + [
        {"id": "e5", "kind": "fill", "label": "Guests", "role": "textbox", "value": "", "node": 4},
    ]
    plan = {"requirements": [{"what": "city", "value": "Lisbon"}], "open": None, "finish": "Seen."}
    with pytest.raises(ValueError, match="Invalid Laya"):
        policy(plan).choose(page(form), [])


def test_a_target_that_never_executes_is_dropped(fake):
    p = policy({"requirements": [], "open": "Casa Flora", "finish": "Casa Flora is open."})
    first = p.choose(page(FORM), [])
    assert first["choice"] == "e4"
    p.choose(page(FORM), [])  # The covered click raised StalePage; history did not grow.
    third = p.choose(page(FORM), [])
    assert third["choice"] != "e4"


@pytest.mark.parametrize(
    ("value", "current", "expected"),
    [
        ("October 20, 2026", "Tue, Oct 20", True),
        ("October 20, 2026", "Wed, Oct 21", False),
        ("Zurich", "Zürich", True),
        ("London", "", False),
        ("one-way", "Round trip", None),
    ],
)
def test_plain_code_settles_what_it_can(value, current, expected):
    element = {"role": "combobox", "current": current, "options": []}
    assert laya.settled({"what": "x", "value": value}, element) is expected


def test_checkbox_requirements_follow_the_checked_state():
    checked = {"role": "checkbox", "current": "checked", "options": []}
    assert laya.settled({"what": "free cancellation", "value": "checked"}, checked)
    assert not laya.settled({"what": "free cancellation", "value": "off"}, checked)


def test_plan_requires_a_finish_condition():
    plan, _ = model.parse_plan({"requirements": [{"what": "to", "value": "London"}], "finish": "Seen."}, {})
    assert plan == {"requirements": [{"what": "to", "value": "London"}], "open": None, "finish": "Seen."}
    with pytest.raises(ValueError, match="no valid plan"):
        model.parse_plan({"requirements": []}, {})


def test_local_text_model_needs_no_key_but_remote_does(monkeypatch):
    monkeypatch.delenv("TEXT_MODEL_API_KEY", raising=False)
    post = Mock(return_value={"choices": [{"message": {"content": '{"text":"Zurich"}'}}]})
    monkeypatch.setattr(model, "post_json", post)
    monkeypatch.setenv("TEXT_MODEL_BASE_URL", "http://localhost:11434/v1")
    assert model.field_text({"goal": "Fly from Zurich"})[0] == "Zurich"
    monkeypatch.setenv("TEXT_MODEL_BASE_URL", "https://openrouter.ai/api/v1")
    with pytest.raises(ValueError, match="TEXT_MODEL_API_KEY"):
        model.field_text({"goal": "Fly from Zurich"})


def test_agent_types_the_planned_value_without_the_text_helper(monkeypatch):
    helper = Mock()
    monkeypatch.setattr(loop, "field_text", helper)
    a = loop.Agent.__new__(loop.Agent)
    a.screenshots, a.pending_text = False, None
    p = page(FORM)
    a.state = {
        "browser": Mock(fresh=Mock(return_value=True), observe=Mock(return_value=p)),
        "page": p, "goal": "g", "history": [], "decisions": [], "status": "predicted",
        "started_at": time.perf_counter(), "record": False, "text_calls": [],
        "decision": {"choice": "e1", "operation": "TYPE_TEXT", "target": "1", "text": "Lisbon", "confidence": 1.0,
                     "probabilities": {"e1": 1.0}, "latency_ms": 1, "usage": {}},
    }
    a.command("act", {"fingerprint": p["fingerprint"]})
    helper.assert_not_called()
    a.state["browser"].act.assert_called_once()
    assert a.state["browser"].act.call_args.kwargs["text"] == "Lisbon"


def test_flight_goal_uses_the_requested_date():
    from examples.flights import goal

    assert "October 20, 2026" in goal(datetime.date(2026, 10, 20))
