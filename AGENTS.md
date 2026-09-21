# Laya Ultrafast

Read README.md before editing. Keep the loop small: page -> indexed elements -> operation + target -> execution.
Google Flights only offers future dates: run `examples/flights.py --date YYYY-MM-DD`.

- The input is one natural-language goal. Do not add site-specific plans or hardcoded field values.
- Local Laya decisions (laya_ultrafast/laya.py) are the default. Laya answers narrow typed questions; generic rules
  compose them. Rules must hold on any site. Measure any rule change on Flights, Wikipedia, and both fixtures.
- The text model plans once per task (requirements, the item to open, the finish condition). In Laya mode TYPE_TEXT
  types a planned value; it does not call the text model per field.
- With DECISION_MODEL=typesafe, TypeSafe chooses an operation and operation-specific target heads in one request.
  Consume only the selected operation's target.
- Targets must map to observed elements and supported operations. Never let the model emit selectors or executable code.
- In TypeSafe mode TYPE_TEXT invokes the text LLM. Cache a stale retry's value only while its entire helper input is identical.
- Never retry a browser mutation. Log execution before observing its result.
- Screenshots are optional; the model does not consume them. Keep demonstration footage at its original speed.
- Keep credentials server-side and .env ignored. Tests must not call paid APIs.
- Verify actual final outcomes independently. A DONE choice is not proof of success.
- Keep examples, README claims, raw evidence, and model-call counts consistent.
- Do not commit or push unless the user requests it.

Checks: uv run ruff check ., uv run pytest, node --check laya_ultrafast/static/app.js, uv build.
