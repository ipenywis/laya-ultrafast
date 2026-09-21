"""Explicit live smoke: local fixtures + paid model APIs. Not run by pytest."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from laya_ultrafast import Agent
from laya_ultrafast.demo import load_environment

GOALS = (
    "Use the destination search and filters to find Design stays in Lisbon with Free cancellation, "
    "then open Casa Flora."
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-actions", type=int, default=15)
    parser.add_argument("--goal", default=GOALS)
    args = parser.parse_args()
    load_environment()
    output = Path("artifacts/dynamic/fixture") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=True)
    print(f"Trace: {output}", flush=True)
    with Agent("http://127.0.0.1:8766/fixture.html?scenario=travel", args.goal) as agent:
        try:
            for state in agent.run():
                history = state["history"]
                print(
                    state["elapsed_ms"], "ms", len(history), "actions",
                    history[-1]["action"] if history else "", flush=True,
                )
                (output / "state.json").write_text(json.dumps(state, indent=2))
                if len(history) >= args.max_actions:
                    raise RuntimeError(f"Diagnostic stopped at {args.max_actions} actions")
        finally:
            state = agent.snapshot()
            try:
                state["verification_text"] = agent.browser.evaluate("document.body.innerText")
            finally:
                (output / "state.json").write_text(json.dumps(state, indent=2))
        assert state["status"] == "done"
        assert state["page"]["url"].endswith("#casa-flora")
        assert "Your filters: Design · Free cancellation enabled · Destination Lisbon" in state["verification_text"]
        result = {
            "ms": state["elapsed_ms"],
            "verified": True,
            "decisions": len(state["decisions"]),
            "actions": len(state["history"]),
        }
        print(json.dumps(result, indent=2))
        (output / "summary.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
