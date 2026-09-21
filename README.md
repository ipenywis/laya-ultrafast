# Laya Ultrafast ⚡

A fork of [browser-use/jev-ultrafast](https://github.com/browser-use/jev-ultrafast) that makes decisions on your own Mac with open weights: free, and offline apart from the websites it browses and an optional text model.

**A browser agent with a dynamic, indexed action space.**

## Local mode: Laya (default)

Decisions now run on your Mac with [Laya](https://github.com/mizorewww/laya-mlx), an open-weight typed-decision model on MLX: no decision API and no per-step cost. The text model (OpenRouter, or a local server such as Ollama) is called **once per task** to turn the goal into field values and a finish condition.

Laya answers narrow questions well: which field sets "destination", whether `Tue, Oct 20` satisfies `October 20, 2026`, which suggestion matches "London". It does not reliably answer the open question "what should the browser do next?" (2 of 5 hand-made Flights steps in a direct Jev-style port). So [laya.py](laya_ultrafast/laya.py) composes narrow questions with rules that apply on any site:

1. After typing or opening a control, choose from the options that appeared.
2. Fill each requirement in the goal's order: map it to a field (Laya, asked both ways), check its value (plain code, then Laya), then type, select, or click.
3. Submit typed forms, then open the item the goal names, or wait for results that name the requested values.

Every target is still an observed element, the executor is unchanged, and there are no site-specific plans.

```bash
uv sync
hf download aac6fef/laya-typed-decisions-mlx   # once; later runs are offline
cp .env.example .env                            # add TEXT_MODEL_API_KEY, or point it at a local server
uv run laya
```

Requires Apple Silicon and macOS 14+. Set `DECISION_MODEL=typesafe` to use the hosted Jev policy described below.

Measured on an M1 Max with `inception/mercury-2.5` on OpenRouter as the text model. Timing includes the ~1–1.5 s planning call:

| Task | Result | Time |
| --- | --- | --- |
| Google Flights, one way Zürich → London, verified by `examples/flights.py --date 2026-10-20` | 5/5 passed | 7.5–12.1 s |
| Wikipedia: open the Gödel's incompleteness theorems article | 3/3 | 3.2–4.6 s |
| Local hotel fixture: filters, search, open Casa Flora | 3/3 | 1.6–1.9 s |
| Local reading-room fixture: open the matching article | 3/3 | ~1 s |

An earlier version of the policy, run fully offline with `gemma4` in Ollama, passed the same tasks. Mercury returns malformed JSON on roughly 1 in 5 planning calls, so the planner retries up to 3 times.

The median Laya decision takes 33 ms. These runs are a small, repeated set of tasks, not a general reliability benchmark.

## Hosted mode: TypeSafe Jev

Give it one goal. [TypeSafe's Jev](https://docs.typesafe.ai/introduction) picks an operation and an element. A small LLM writes text only when the operation is `TYPE_TEXT`.

**Zürich → London on Google Flights in 7.1 seconds.** One natural-language goal, actual text generation, and loading waits included.

<a href="docs/demo.mp4"><img src="docs/demo.gif" alt="A real Google Flights search at 1× speed, with generated city names and dynamic operation/target decisions" width="100%" /></a>

[Watch the MP4](docs/demo.mp4) · [Measurements](docs/performance.md) · [Read the loop](laya_ultrafast/agent.py)

## The action space

Every observation produces a new element table:

```text
[1] button    Change ticket type · Round trip
[2] combobox  Where from?        · San Francisco
[3] combobox  Where to?          · empty
[4] textbox   Departure          · empty
...
```

The operations are `CLICK`, `TYPE_TEXT`, `SELECT`, `SCROLL_UP`, `SCROLL_DOWN`, `WAIT`, `DONE`, and `BLOCKED`. Only supported operations and targets are offered.

```text
                      one TypeSafe request
                     ┌───────────────────────────┐
page → element table → operation                 │
                     │ click_target              │
                     │ type_text_target          │
                     │ select_target, if present │
                     └─────────────┬─────────────┘
                         use the matching target
                                   │
                    CLICK [7] ─────┤──→ browser
                TYPE_TEXT [3] ─────┘
                          ↓
                   small LLM → text → browser
```

Target questions are speculative. If the operation is `CLICK`, only `click_target` can execute. Two decisions, **one network round trip**. Each target head contains only compatible elements. Native dropdown choices carry an observed element/option index.

There are no site-specific action scripts or prepared field strings in the policy. The Flights example supplies a goal and independently verifies the outcome. The screenshot renderer adds labels afterward; it does not drive the browser.

## Try it

```bash
cd laya-ultrafast
uv sync
cp .env.example .env
# Add TEXT_MODEL_API_KEY. For hosted decisions also set DECISION_MODEL=typesafe and TYPESAFE_API_KEY.
uv run laya
```

Open **http://127.0.0.1:8766** and click **Start demo → Run automatically**. The inspector shows numbered elements, operation probabilities, target probabilities, and executed actions. **Choose next** pauses before execution.

Chrome connects through [Browser Harness](https://github.com/browser-use/browser-harness), installed by `uv sync`. Run `uv run browser-harness --doctor` if it needs connecting. Allow remote debugging in Chrome when prompted.

`TEXT_MODEL_API_KEY` is an OpenRouter key in the example configuration. The current demo uses `inception/mercury-2.5` with reasoning disabled. Gemini, GLM, and DeepSeek can also use the OpenAI-compatible text helper; configure the appropriate model, endpoint, and reasoning setting.

## Use the library

```python
from laya_ultrafast import Agent

with Agent(
    "https://www.google.com/travel/flights?hl=en",
    "Find one-way flights from Zurich to London on October 20, 2026, "
    "for one adult in economy. Stop when matching flight options are visible.",
) as agent:
    for state in agent.run():
        print(state["elapsed_ms"], state["status"])
```

Run with `uv run --env-file .env python your_script.py`. The same policy can run a different task:

```bash
uv run --env-file .env python examples/run.py \
  --url https://en.wikipedia.org/wiki/Main_Page \
  --goal 'Find and open the Wikipedia article about Gödel’s incompleteness theorems.'
```

`uv run --env-file .env python examples/flights.py --date 2026-10-20 --keep-open` performs the flight search, checks the actual route/date/results, and saves its trace. It does not select or book a flight.

## Why it moves

- **One request per decision cycle.** Operation and target heads share the same observed state.
- **No screenshots in the default agent loop.** Jev consumes structured state. The inspector opts into screenshots; the video uses a separate continuous screencast.
- **One browser call per snapshot.** Read visible controls, their names, values, and text atomically. Keep references to the actual DOM nodes.
- **Validate the selected target.** Clicks check the document, form values, target, and nearby context. Animation alone does not force another prediction. Resolve current geometry and reject covered controls before input.
- **Wait for useful state.** After typing into a combobox, wait for visible suggestions, capped at 200 ms. Other interactions get at most two animation frames or 50 ms. These reads happen after execution is logged.
- **Keep hidden tabs rendering.** Focus emulation prevents background animation throttling without switching Chrome's visible tab.
- **Send visible text.** Offscreen article bodies and footers do not fill the model context.
- **Reuse an interrupted text request.** A generated value survives a stale-page retry only if the entire text-helper input is unchanged.

Every executed target is resolved from an observed node. The executor rechecks page freshness and click occlusion. Model output never becomes selectors, coordinates, shell commands, or executable JavaScript. Text-helper output must parse as a small JSON object before typing.

## Small enough to read

| File | Job |
| --- | --- |
| [agent.py](laya_ultrafast/agent.py) | The complete loop and text-helper handoff |
| [snapshot.js](laya_ultrafast/snapshot.js) | Atomic DOM snapshot, indexed controls, freshness guards |
| [browser.py](laya_ultrafast/browser.py) | Browser connection, current geometry, execution |
| [model.py](laya_ultrafast/model.py) | Dynamic operation/target heads and text generation |
| [questions.py](laya_ultrafast/questions.py) | Model instructions |
| [demo.py](laya_ultrafast/demo.py) | Local inspector |

## Evidence and limits

The current video is a **7,073 ms** Google Flights run. Timing starts after initial page observation and includes model calls, generated text, browser work, stale decisions, and loading waits. A fresh independent check verifies the one-way setting, Zürich, London, September 20, 2026, and visible flight options. The video plays at 1×, with no opening hold and a 0.5-second final hold.

In six alternating runs with identical models and settings, both versions passed **3/3**. Median task time went from **9.450 s → 7.092 s**, a **25% reduction**; median browser protocol calls went from **1,092 → 101**. This is three repeats of one task on one browser profile, not a general reliability benchmark.

The same policy opened the requested Wikipedia article in **2.798 s** and passed a local hotel search/filter task in **1.896 s**. Runs, failures, source hashes, and measurement boundaries are in [performance.md](docs/performance.md).

A `DONE` choice still requires independent outcome verification. The DOM reader handles common HTML and ARIA controls, not the full accessible-name specification. Shadow roots, frames, canvas, uploads, pop-up tabs, nested scrolling, and arbitrary keyboard widgets remain outside this MVP. Owned tabs share the existing Chrome profile.

## Development

```bash
uv run ruff check .
uv run pytest
node --check laya_ultrafast/static/app.js
node --check laya_ultrafast/snapshot.js
uv build
```

Tests are offline. `uv run python scripts/check_guards.py` checks real controls in a local browser without model calls. Live examples and recording scripts make paid API calls. `scripts/record_flights.py <new-folder>` captures original browser timestamps; `scripts/render_demo.py <recording-folder>` renders that verified run at 1× and crops out the Google account strip. Credentials and raw traces stay ignored.

---

[Browser Use](https://github.com/browser-use/browser-use) · [Browser Harness](https://github.com/browser-use/browser-harness) · [TypeSafe speculative fan-out](https://docs.typesafe.ai/patterns/fan-out)
