"""CLI-backed local model adapter for Codex and Claude CLI.

Provides direct chat execution, model resolution, and an ephemeral
OpenAI-compatible HTTP server endpoint for text planning without API keys.
"""

import difflib
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from collections.abc import Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Literal

ModelProvider = Literal["CODEX", "CLAUDE"]

DEFAULT_CODEX_MODELS = (
    "gpt-6-luna",
    "gpt-6-sol",
    "gpt-6-astra",
)

DEFAULT_CLAUDE_MODELS = (
    "sonnet",
    "haiku",
    "opus",
    "fable",
)

CODEX_MODELS = DEFAULT_CODEX_MODELS
CLAUDE_MODELS = DEFAULT_CLAUDE_MODELS


def get_default_codex_model() -> str:
    config_file = Path.home() / ".codex" / "config.toml"
    if config_file.exists():
        try:
            for line in config_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("model") and "=" in line:
                    val = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if val:
                        return val
        except Exception:
            pass
    return "gpt-6-luna"


def discover_codex_models() -> list[str]:
    """Dynamically discover models configured in environment or ~/.codex/config.toml."""
    discovered: list[str] = []
    env_models = os.environ.get("CODEX_MODELS")
    if env_models:
        discovered.extend([m.strip() for m in env_models.split(",") if m.strip()])

    config_file = Path.home() / ".codex" / "config.toml"
    if config_file.exists():
        try:
            content = config_file.read_text(encoding="utf-8")
            for line in content.splitlines():
                line = line.strip()
                if line.startswith("model") and "=" in line:
                    val = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if val and val not in discovered:
                        discovered.append(val)
            in_section = False
            for line in content.splitlines():
                line = line.strip()
                if line.startswith("[") and ("model" in line.lower() or "tui" in line.lower()):
                    in_section = True
                    continue
                elif line.startswith("["):
                    in_section = False
                if (
                    in_section
                    and "=" in line
                    and not line.startswith("#")
                ):
                    key = line.split("=", 1)[0].strip().strip('"').strip("'")
                    if (
                        key
                        and not key.startswith("followUp")
                        and not key.startswith("model_")
                        and key not in discovered
                    ):
                        discovered.append(key)
        except Exception:
            pass

    for default in DEFAULT_CODEX_MODELS:
        if default not in discovered:
            discovered.append(default)

    return discovered


def discover_claude_models() -> list[str]:
    """Dynamically discover models configured in environment or ~/.claude.json."""
    discovered: list[str] = []
    env_models = os.environ.get("CLAUDE_MODELS")
    if env_models:
        discovered.extend([m.strip() for m in env_models.split(",") if m.strip()])

    claude_json = Path.home() / ".claude.json"
    if claude_json.exists():
        try:
            data = json.loads(claude_json.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                val = data.get("model") or data.get("defaultModel")
                if isinstance(val, str) and val.strip() and val not in discovered:
                    discovered.append(val.strip())
        except Exception:
            pass

    for default in DEFAULT_CLAUDE_MODELS:
        if default not in discovered:
            discovered.append(default)

    return discovered


def get_model_candidates(provider: ModelProvider) -> list[str]:
    """Return candidates for a provider, dynamically discovering from config and environment."""
    if provider == "CODEX":
        return discover_codex_models()
    elif provider == "CLAUDE":
        return discover_claude_models()
    return []


def tokenize_model_name(name: str) -> set[str]:
    """Extract normalized component tokens from a model name."""
    return {
        token
        for token in re.split(r"[-_.\s/]+", name.lower())
        if token and token not in ("claude", "gpt", "anthropic", "openai", "models")
    }


def model_similarity(query: str, candidate: str) -> float:
    """Calculate generic similarity score between query and a candidate model name."""
    q_lower, c_lower = query.lower().strip(), candidate.lower().strip()
    if q_lower == c_lower:
        return 1.0

    q_norm = re.sub(r"[-_.\s]+", "-", q_lower)
    c_norm = re.sub(r"[-_.\s]+", "-", c_lower)
    if q_norm == c_norm:
        return 0.99

    if q_norm in c_norm or c_norm in q_norm:
        return 0.90

    q_tokens = tokenize_model_name(q_lower)
    c_tokens = tokenize_model_name(c_lower)
    if q_tokens and c_tokens:
        shared = q_tokens & c_tokens
        if shared:
            jaccard = len(shared) / len(q_tokens | c_tokens)
            if shared == q_tokens or shared == c_tokens:
                return 0.80 + 0.15 * jaccard
            return 0.60 + 0.20 * jaccard

    return difflib.SequenceMatcher(None, q_norm, c_norm).ratio()


def get_model_provider(val: str | None = None) -> ModelProvider | None:
    """Read and validate the model_provider typed environment variable."""
    if val is None:
        val = (
            os.environ.get("MODEL_PROVIDER")
            or os.environ.get("model_provider")
            or os.environ.get("TEXT_MODEL_PROVIDER")
        )
    if not val:
        return None
    normalized = val.strip().upper()
    if normalized in ("CODEX", "CLAUDE"):
        return normalized  # type: ignore[return-value]
    raise ValueError(f"Invalid model_provider '{val}'. Must be 'CODEX' or 'CLAUDE'.")


def find_best_model_match(
    provider: ModelProvider,
    query: str | None = None,
    candidates: Sequence[str] | None = None,
) -> str:
    """Find the best matching model for the chosen CLI provider.

    If no query is specified, returns the provider's default model.
    Matches exact names, token subsets, substrings, or closest fuzzy matches.
    If no candidate sufficiently matches, returns the query string as-is.
    """
    if candidates is None:
        cands = get_model_candidates(provider)
    else:
        cands = list(candidates)

    if not query or not query.strip():
        return cands[0] if cands else (get_default_codex_model() if provider == "CODEX" else "sonnet")

    q = query.strip()
    scored = [(model_similarity(q, c), c) for c in cands]
    scored.sort(key=lambda x: x[0], reverse=True)

    if scored and scored[0][0] >= 0.60:
        return scored[0][1]

    return q


def extract_json_object(raw: str) -> dict:
    """Extract and parse a JSON dictionary from CLI model text output."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        try:
            data = json.loads(text[first_brace : last_brace + 1])
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    raise ValueError(f"Could not parse valid JSON object from model output: {raw[:200]}")


def run_cli_chat(
    provider: ModelProvider,
    model: str,
    system: str,
    prompt_content: str,
    timeout: int = 120,
) -> dict:
    """Execute a text chat completion through Codex or Claude CLI."""
    if provider == "CODEX":
        if not shutil.which("codex"):
            raise RuntimeError("The 'codex' CLI is not found in PATH.")
        prompt = (
            f"{system}\n\nInput:\n{prompt_content}\n\n"
            "Respond ONLY with a valid JSON object matching the requested schema. "
            "Do not include markdown codeblocks or commentary."
        )
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            output_path = f.name
        try:
            cmd = [
                "codex",
                "exec",
                "--ephemeral",
                "-s",
                "read-only",
                "--ignore-rules",
                "-m",
                model,
                "-o",
                output_path,
                prompt,
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            if res.returncode != 0:
                err = (res.stderr or res.stdout or "").strip()
                raise RuntimeError(f"Codex CLI failed (exit {res.returncode}): {err[:300]}")
            with open(output_path, "r", encoding="utf-8") as f:
                content = f.read()
            return extract_json_object(content)
        finally:
            if os.path.exists(output_path):
                try:
                    os.unlink(output_path)
                except OSError:
                    pass

    elif provider == "CLAUDE":
        if not shutil.which("claude"):
            raise RuntimeError("The 'claude' CLI is not found in PATH.")
        prompt = (
            f"{prompt_content}\n\n"
            "Respond ONLY with a valid JSON object matching the requested schema. "
            "Do not include markdown codeblocks or commentary."
        )
        cmd = [
            "claude",
            "-p",
            "--tools",
            "",
            "--system-prompt",
            system,
            "--model",
            model,
            prompt,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if res.returncode != 0:
            err = (res.stderr or res.stdout or "").strip()
            raise RuntimeError(f"Claude CLI failed (exit {res.returncode}): {err[:300]}")
        return extract_json_object(res.stdout)

    raise ValueError(f"Unsupported model provider: {provider}")


class AdapterHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_POST(self):
        if self.path.rstrip("/") not in ("/v1/chat/completions", "/chat/completions"):
            self.send_response(404)
            self.end_headers()
            return

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8")
        try:
            req_data = json.loads(body)
            messages = req_data.get("messages", [])
            system = ""
            user_content = ""
            for m in messages:
                if m.get("role") == "system":
                    system += str(m.get("content", "")) + "\n"
                elif m.get("role") == "user":
                    user_content += str(m.get("content", "")) + "\n"

            provider = getattr(self.server, "provider", "CODEX")
            model = getattr(self.server, "model", "default")
            parsed = run_cli_chat(provider, model, system.strip(), user_content.strip())
            resp_body = {
                "id": "chatcmpl-local-adapter",
                "object": "chat.completion",
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(parsed),
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }
            resp_bytes = json.dumps(resp_body).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp_bytes)))
            self.end_headers()
            self.wfile.write(resp_bytes)
        except Exception as e:
            err_body = json.dumps({"error": {"message": str(e)}}).encode("utf-8")
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(err_body)))
            self.end_headers()
            self.wfile.write(err_body)


class CLIModelAdapter:
    """A reusable local adapter for running text models via the Codex or Claude CLI.

    Can execute chat completions directly or host a temporary loopback HTTP server
    compatible with the OpenAI /v1/chat/completions API.
    """

    def __init__(
        self,
        provider: ModelProvider | str | None = None,
        model_name: str | None = None,
        candidates: Sequence[str] | None = None,
        host: str = "127.0.0.1",
        port: int = 0,
    ):
        self.provider: ModelProvider = (
            get_model_provider(provider) if provider else (get_model_provider() or "CODEX")
        )
        raw_name = (
            model_name
            or os.environ.get("MODEL_PROVIDER_NAME")
            or os.environ.get("model_provider_name")
            or os.environ.get("TEXT_MODEL")
        )
        self.candidates = list(candidates) if candidates is not None else get_model_candidates(self.provider)
        self.model = find_best_model_match(self.provider, raw_name, candidates=self.candidates)
        self.host = host
        self.port = port
        self.server: ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None
        self.endpoint_url: str | None = None
        self._prev_base_url: str | None = None
        self._prev_model: str | None = None

    @classmethod
    def from_env(cls) -> "CLIModelAdapter | None":
        """Instantiate an adapter if MODEL_PROVIDER or model_provider is configured in env."""
        provider = get_model_provider()
        if not provider:
            return None
        return cls(provider=provider)

    def chat(self, system: str, context: dict | str, timeout: int = 120) -> tuple[dict, dict]:
        """Execute a structured JSON chat completion through the CLI."""
        prompt_content = json.dumps(context) if isinstance(context, dict) else str(context)
        started = time.perf_counter()
        output = run_cli_chat(self.provider, self.model, system, prompt_content, timeout=timeout)
        meta = {
            "model": f"{self.provider.lower()}:{self.model}",
            "latency_ms": round((time.perf_counter() - started) * 1000),
            "usage": {"provider": self.provider, "model": self.model},
        }
        return output, meta

    def start(self) -> str:
        """Start the background HTTP server and return its base URL."""
        self.server = ThreadingHTTPServer((self.host, self.port), AdapterHandler)
        self.server.provider = self.provider  # type: ignore[attr-defined]
        self.server.model = self.model  # type: ignore[attr-defined]
        actual_port = self.server.server_port
        self.endpoint_url = f"http://{self.host}:{actual_port}/v1"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return self.endpoint_url

    def stop(self) -> None:
        """Stop the background HTTP server."""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            self.server = None
        if self.thread:
            self.thread.join(timeout=2)
            self.thread = None
        self.endpoint_url = None

    def __enter__(self) -> str:
        url = self.start()
        self._prev_base_url = os.environ.get("TEXT_MODEL_BASE_URL")
        self._prev_model = os.environ.get("TEXT_MODEL")
        os.environ["TEXT_MODEL_BASE_URL"] = url
        os.environ["TEXT_MODEL"] = self.model
        return url

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        if self._prev_base_url is not None:
            os.environ["TEXT_MODEL_BASE_URL"] = self._prev_base_url
        else:
            os.environ.pop("TEXT_MODEL_BASE_URL", None)
        if self._prev_model is not None:
            os.environ["TEXT_MODEL"] = self._prev_model
        else:
            os.environ.pop("TEXT_MODEL", None)


# Alias for backwards compatibility
TemporaryLocalAdapter = CLIModelAdapter


def create_text_planning_endpoint(
    provider: ModelProvider | str | None = None,
    model_name: str | None = None,
    candidates: Sequence[str] | None = None,
    host: str = "127.0.0.1",
    port: int = 0,
) -> CLIModelAdapter:
    """Create a temporary local adapter endpoint providing text planning through Codex or Claude CLI."""
    return CLIModelAdapter(
        provider=provider,
        model_name=model_name,
        candidates=candidates,
        host=host,
        port=port,
    )
