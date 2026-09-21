"""Live Skyscanner search. Same goal shape and policy as examples/flights.py; never selects or books a flight.

Skyscanner may show an "Are you a person or a robot?" check, especially to automated or headless browsers.
The agent does not try to get past it: solve it yourself in the tab, or run in your everyday Chrome profile.
"""

import argparse
import datetime
import json
import re
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from laya_ultrafast import Agent

URL = "https://www.skyscanner.net/"


def goal(day):
    # Skyscanner ticks "Add a place to stay" by default, which turns the search into a hotel search.
    # The goal says so in words; the policy has no Skyscanner-specific rule.
    return (
        f"Find one-way flights from Zurich to London on {day:%B} {day.day}, {day.year}, for one adult in economy, "
        "without adding a place to stay. Stop when matching flight options are visible. "
        "Do not select or book a flight."
    )


def verify(page, day):
    """Independent checks on the final page: route, date and one-way in the search URL, and visible fares."""
    parsed = urlparse(page["url"])
    parts = [p for p in parsed.path.lower().split("/") if p]
    query = parse_qs(parsed.query)
    route = parts[2:5] if parts[:2] == ["transport", "flights"] else []
    checks = {
        "search_page": parsed.hostname is not None and "skyscanner" in parsed.hostname and len(route) == 3,
        "origin": bool(route) and route[0] in {"zrh", "zurh", "zuri"},
        "destination": bool(route) and route[1] in {"lond", "lon", "lhr", "lgw", "lcy", "stn", "ltn", "sen"},
        "date": bool(route) and route[2] == f"{day:%y%m%d}",
        "one_way": query.get("rtn") == ["0"] or len(parts) == 5,
        "not_captcha": "captcha" not in page["url"],
        "results": len(re.findall(r"(?:£|€|\$|CHF|DZD|د\.ج)\s?[\d,.]+|[\d,.]+\s?(?:CHF|DZD|د\.ج)", page["text"])) >= 2,
    }
    return {"passed": all(checks.values()), "checks": checks}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/skyscanner/latest")
    parser.add_argument("--keep-open", action="store_true")
    parser.add_argument(
        "--date", type=datetime.date.fromisoformat, default=datetime.date.today() + datetime.timedelta(days=30),
        help="Departure date, YYYY-MM-DD. Defaults to 30 days ahead.",
    )
    args = parser.parse_args()
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)
    agent = Agent(URL, goal(args.date))
    try:
        for state in agent.run():
            last = state["history"][-1] if state["history"] else {}
            print(state["elapsed_ms"], state["status"], last.get("action", ""), flush=True)
    finally:
        state = agent.snapshot()
        state["verification"] = verify(state["page"], args.date)
        (folder / "state.json").write_text(json.dumps(state, indent=2))
        if not args.keep_open:
            agent.close()
    print(json.dumps(state["verification"], indent=2))
    if not state["verification"]["passed"]:
        raise SystemExit("Final page did not satisfy the route/date checks")


if __name__ == "__main__":
    main()
