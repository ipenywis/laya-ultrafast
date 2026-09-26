"""Unit tests for the CLI model adapters and TemporaryLocalAdapter endpoint. No paid APIs."""

import json
from unittest.mock import Mock

import httpx
import pytest

from laya_ultrafast import adapter, model
from laya_ultrafast.adapter import CLIModelAdapter


def test_get_model_provider():
    assert adapter.get_model_provider("CODEX") == "CODEX"
    assert adapter.get_model_provider("codex") == "CODEX"
    assert adapter.get_model_provider("CLAUDE") == "CLAUDE"
    assert adapter.get_model_provider("claude") == "CLAUDE"
    assert adapter.get_model_provider("") is None

    with pytest.raises(ValueError, match="Invalid model_provider"):
        adapter.get_model_provider("UNSUPPORTED")


def test_get_model_provider_from_env(monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "codex")
    assert adapter.get_model_provider() == "CODEX"

    monkeypatch.setenv("MODEL_PROVIDER", "claude")
    assert adapter.get_model_provider() == "CLAUDE"

    monkeypatch.delenv("MODEL_PROVIDER", raising=False)
    monkeypatch.setenv("model_provider", "CODEX")
    assert adapter.get_model_provider() == "CODEX"


def test_find_best_model_match():
    # Claude matching
    assert adapter.find_best_model_match("CLAUDE", "sonnet") == "sonnet"
    assert adapter.find_best_model_match("CLAUDE", "haiku") == "haiku"
    assert adapter.find_best_model_match("CLAUDE", "claude-3.5-sonnet") == "sonnet"
    assert adapter.find_best_model_match("CLAUDE", None) == "sonnet"
    assert adapter.find_best_model_match("CLAUDE", "custom-model-x") == "custom-model-x"

    # Codex matching
    assert adapter.find_best_model_match("CODEX", "luna") == "gpt-6-luna"
    assert adapter.find_best_model_match("CODEX", "sol") == "gpt-6-sol"
    assert adapter.find_best_model_match("CODEX", "astra") == "gpt-6-astra"
    assert adapter.find_best_model_match("CODEX", "gpt-6-sol") == "gpt-6-sol"

    # Custom updated candidates list
    custom_cands = ["gpt-7-future", "gpt-6-terra", "claude-4-ultra"]
    assert adapter.find_best_model_match("CODEX", "terra", candidates=custom_cands) == "gpt-6-terra"
    assert adapter.find_best_model_match("CODEX", "future", candidates=custom_cands) == "gpt-7-future"
    assert adapter.find_best_model_match("CLAUDE", "ultra", candidates=custom_cands) == "claude-4-ultra"



def test_extract_json_object():
    # Raw JSON
    assert adapter.extract_json_object('{"text": "val"}') == {"text": "val"}

    # Markdown code fence
    assert adapter.extract_json_object('```json\n{"text": "val"}\n```') == {"text": "val"}

    # Text wrapping JSON
    assert adapter.extract_json_object('Result: {"text": "val"}\nDone.') == {"text": "val"}

    with pytest.raises(ValueError, match="Could not parse valid JSON object"):
        adapter.extract_json_object("Not a json at all")


def test_run_cli_chat_codex(monkeypatch, tmp_path):
    monkeypatch.setattr(adapter.shutil, "which", lambda cmd: "/usr/local/bin/codex")

    def mock_subprocess_run(cmd, capture_output=True, text=True, timeout=120):
        # cmd has -o <output_path>
        out_idx = cmd.index("-o") + 1
        with open(cmd[out_idx], "w", encoding="utf-8") as f:
            f.write('{"requirements": [{"what": "search", "value": "test"}], "open": null, "finish": "done"}')
        return Mock(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(adapter.subprocess, "run", mock_subprocess_run)

    result = adapter.run_cli_chat("CODEX", "gpt-6-sol", "system", "input")
    assert result["finish"] == "done"
    assert result["requirements"][0]["value"] == "test"


def test_run_cli_chat_claude(monkeypatch):
    monkeypatch.setattr(adapter.shutil, "which", lambda cmd: "/usr/local/bin/claude")

    def mock_subprocess_run(cmd, capture_output=True, text=True, timeout=120):
        return Mock(returncode=0, stdout='```json\n{"text": "Zurich"}\n```', stderr="")

    monkeypatch.setattr(adapter.subprocess, "run", mock_subprocess_run)

    result = adapter.run_cli_chat("CLAUDE", "sonnet", "system", "input")
    assert result == {"text": "Zurich"}


def test_cli_model_adapter_class(monkeypatch):
    monkeypatch.setattr(
        adapter,
        "run_cli_chat",
        lambda provider, model_name, sys, prompt, timeout=120: {"text": "Paris"},
    )

    inst = CLIModelAdapter(provider="CLAUDE", model_name="sonnet")
    output, meta = inst.chat("sys", {"goal": "Paris"})
    assert output == {"text": "Paris"}
    assert meta["model"] == "claude:sonnet"


def test_temporary_local_adapter_http_endpoint(monkeypatch):
    monkeypatch.setattr(
        adapter,
        "run_cli_chat",
        lambda provider, model_name, sys, prompt: {"requirements": [], "open": None, "finish": "ready"},
    )

    with CLIModelAdapter(provider="CODEX", model_name="luna") as endpoint:
        assert endpoint.startswith("http://127.0.0.1:")
        client = httpx.Client(timeout=5)
        resp = client.post(
            f"{endpoint}/chat/completions",
            json={"messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "usr"}]},
        )
        assert resp.status_code == 200
        data = resp.json()
        content = json.loads(data["choices"][0]["message"]["content"])
        assert content["finish"] == "ready"


def test_chat_json_with_model_provider(monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "CLAUDE")
    monkeypatch.setenv("MODEL_PROVIDER_NAME", "sonnet")

    mock_chat = Mock(return_value=({"text": "Geneva"}, {"model": "claude:sonnet"}))
    monkeypatch.setattr(CLIModelAdapter, "chat", mock_chat)

    output, meta = model.chat_json("system prompt", {"goal": "Fly to Geneva"})
    assert output == {"text": "Geneva"}
    assert meta["model"] == "claude:sonnet"
    assert mock_chat.call_count == 1
