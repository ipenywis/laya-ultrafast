# Laya Ultrafast ⚡

**A local, open-weight port of [browser-use/jev-ultrafast](https://github.com/browser-use/jev-ultrafast).**

> [!NOTE]
> This project is a clone of **[jev-ultrafast](https://github.com/browser-use/jev-ultrafast) by [Browser Use](https://github.com/browser-use)**, ported to make its decisions with **[Laya](https://github.com/mizorewww/laya-mlx)** running locally through MLX. The browser agent, DOM snapshot, executor, safety checks, inspector and most of the design are theirs. All credit for the original work goes to the jev-ultrafast authors. For how the agent works, see the [original repository](https://github.com/browser-use/jev-ultrafast).

> [!IMPORTANT]
> **Apple Silicon only.** Laya runs through [laya-mlx](https://github.com/mizorewww/laya-mlx), which needs an M-series Mac, macOS 14+, and Python 3.11+ (this project uses 3.12+).

## What is different from jev-ultrafast

jev-ultrafast asks [TypeSafe's Jev](https://docs.typesafe.ai/introduction), a hosted model, to choose each browser action. This port replaces that API call with **Laya**, an open-weight typed-decision model that runs on your Mac:

- **No decision API and no per-step cost.** Each decision is a local forward pass. The median is about 33 ms on an M1 Max.
- **One text-model call per task.** An OpenAI-compatible model (OpenRouter by default, or a local server such as Ollama) turns the goal into field values, the item to open, and a finish condition. If you use a local text model, the whole agent runs offline, apart from the websites it browses.
- **A different policy.** Laya answers narrow questions well: which field is the destination, whether `Tue, Oct 20` matches `October 20, 2026`, which suggestion is London. It does not reliably answer the open question "what should the browser do next?". So [`laya_ultrafast/laya.py`](laya_ultrafast/laya.py) combines narrow Laya questions with rules that apply on any site:
  1. Fill the values the goal states. Laya maps each one to a field, and Laya or plain code checks it.
  2. After typing or opening a control, choose from the options that appeared.
  3. Submit, then open the item the goal names, or wait for results that name the requested values.

  Every target is still an element the agent observed on the page, and there are no site-specific plans.
- **Hosted mode still works.** Set `DECISION_MODEL=typesafe` to use the original Jev policy unchanged.

## Laya setup

Laya itself is documented in **[mizorewww/laya-mlx](https://github.com/mizorewww/laya-mlx)**, an independent MLX port of [Convai Innovations' Laya](https://github.com/NandhaKishorM/laya). Read it for requirements, checkpoints, benchmarks and troubleshooting.

This project installs `laya-mlx` as a dependency and uses the **`aac6fef/laya-typed-decisions-mlx`** checkpoint (421M parameters, 1,024-token context). Download it once:

```bash
uv sync                                          # installs laya-mlx and the `hf` command
uv run hf download aac6fef/laya-typed-decisions-mlx
```

Later runs load it from the Hugging Face cache and need no network for decisions. To use another checkpoint, set `LAYA_MODEL`. The laya-mlx README lists the options and their limits: for example, the English `laya` checkpoint has only a 512-token context.

## Quick start

```bash
git clone <this repository>
cd laya-ultrafast
uv sync
uv run hf download aac6fef/laya-typed-decisions-mlx
cp .env.example .env    # then set TEXT_MODEL_API_KEY, or point TEXT_MODEL_BASE_URL at a local server
uv run laya
```

Open **http://127.0.0.1:8766**, choose a scenario, and click **Start demo → Run automatically**.

Chrome connects through [Browser Harness](https://github.com/browser-use/browser-harness), just as in jev-ultrafast. Enable remote debugging at `chrome://inspect/#remote-debugging`, and run `uv run browser-harness --doctor` if the connection fails.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `DECISION_MODEL` | `laya` | `laya` for local decisions, `typesafe` for the original hosted Jev policy |
| `LAYA_MODEL` | `aac6fef/laya-typed-decisions-mlx` | Laya checkpoint, from the Hub or a local path |
| `TEXT_MODEL_BASE_URL` | `https://openrouter.ai/api/v1` | Any OpenAI-compatible endpoint. `localhost` endpoints need no key |
| `TEXT_MODEL` | `inception/mercury-2.5` | Model that plans the task once per run |
| `TEXT_MODEL_API_KEY` | — | Required for remote endpoints |
| `TEXT_MODEL_REASONING` | `none` | Turns reasoning off for faster planning |
| `TYPESAFE_API_KEY`, `TYPESAFE_MODEL` | — | Only for `DECISION_MODEL=typesafe` |

For a fully offline setup with [Ollama](https://ollama.com):

```bash
TEXT_MODEL_BASE_URL=http://localhost:11434/v1
TEXT_MODEL=gemma4:latest
```

## Demos

| Scenario | Command |
| --- | --- |
| Google Flights (checks the final page) | `uv run --env-file .env python examples/flights.py --date 2026-10-20 --keep-open` |
| Skyscanner (checks the final page) | `uv run --env-file .env python examples/skyscanner.py --date 2026-10-20 --keep-open` |
| Any site and goal | `uv run --env-file .env python examples/run.py --url URL --goal 'A narrow goal'` |

Flight sites only offer future dates, so pass `--date`. It defaults to 30 days ahead. The examples never select or book a flight.

**Skyscanner** may show an "Are you a person or a robot?" check, especially to automated or headless browsers. The agent does not try to get past it. Run it in your everyday Chrome and solve the check yourself if it appears. Skyscanner also ticks "Add a place to stay" by default, so its goal says "without adding a place to stay".

## Measurements

These were measured on an M1 Max with `inception/mercury-2.5` on OpenRouter as the text model. Timing includes the planning call (~1–1.5 s):

| Task | Result | Time |
| --- | --- | --- |
| Google Flights, one way Zürich → London, checked by `examples/flights.py` | 5/5 passed | 7.5–12.1 s |
| Wikipedia: open the Gödel's incompleteness theorems article | 2/2 | ~3–5 s |
| Local hotel fixture: filters, search, open Casa Flora | 2/2 | ~1.7 s |
| Local reading-room fixture: open the matching article | 2/2 | ~1 s |
| Skyscanner | Form filled end to end, then blocked by the robot check in headless testing | — |

This is a small set of repeated tasks, not a general reliability benchmark. The original Jev measurements, video and methodology are in [jev-ultrafast](https://github.com/browser-use/jev-ultrafast).

## Limitations

- Runs only on Apple Silicon, because Laya runs through MLX.
- The Laya policy is new and tested on few sites. The original jev-ultrafast limits still apply: shadow roots, frames, canvas, uploads, pop-up tabs, nested scrolling and arbitrary keyboard widgets are out of scope. See [its README](https://github.com/browser-use/jev-ultrafast#evidence-and-limits).
- A `DONE` decision is not proof of success. The examples check the final page independently.
- Planning quality depends on the text model. `inception/mercury-2.5` sometimes returns malformed JSON, so the planner retries up to 3 times.

## Everything else

The action space, DOM snapshot, executor, freshness and occlusion checks, the inspector, and the performance work all come from jev-ultrafast. **See [browser-use/jev-ultrafast](https://github.com/browser-use/jev-ultrafast)** for how they work, the design notes, and the original evidence. The files in [`docs/`](docs/) are the original project's records and describe the hosted Jev runs.

Development checks are the same as upstream:

```bash
uv run ruff check .
uv run pytest            # offline: a fake stands in for Laya; no downloads or paid calls
node --check laya_ultrafast/static/app.js
node --check laya_ultrafast/snapshot.js
uv build
```

## Credits

- **[jev-ultrafast](https://github.com/browser-use/jev-ultrafast)** by [Browser Use](https://github.com/browser-use): the original project. This repository is a clone and a port of it.
- **[Browser Harness](https://github.com/browser-use/browser-harness)** by Browser Use: the Chrome connection.
- **[laya-mlx](https://github.com/mizorewww/laya-mlx)**: the MLX runtime and converted checkpoints for Laya.
- **[Laya](https://github.com/NandhaKishorM/laya)** by Convai Innovations and contributors: the model and its weights.
- **[TypeSafe Jev](https://docs.typesafe.ai/introduction)**: the hosted policy the original project uses, still available here as `DECISION_MODEL=typesafe`.

## License

[MIT](LICENSE), unchanged from the original: Copyright (c) 2026 Browser Use. Laya and laya-mlx are Apache-2.0 under their own licenses. See their repositories.
