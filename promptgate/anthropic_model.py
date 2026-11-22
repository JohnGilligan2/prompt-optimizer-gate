"""Optional live adapter: the `model(system, user, schema) -> dict` callable on
the Anthropic API, with schema-enforced JSON output.

Not imported by the package or the tests; `pip install anthropic` to use it.

    from promptgate.anthropic_model import AnthropicModel
    model = AnthropicModel(model="<a current model id>")
    run_cycle(..., model=model, ...)

Two habits from production worth keeping:

  * No temperature. The current top-tier models reject sampling parameters on
    this path, and determinism was never going to come from temperature=0
    anyway. It comes from the schema, and from replaying in the gate.
  * A verdict cut off by the token ceiling is still "valid JSON so far", so
    check stop_reason BEFORE parsing and say "truncated" instead of reporting
    a confusing parse error.
"""

from __future__ import annotations

import json
import os


class ModelUnusable(RuntimeError):
    """The response cannot be used: truncated, refused, or no text."""


class AnthropicModel:
    def __init__(self, model: str, max_tokens: int = 8000, api_key: str | None = None):
        import anthropic  # deferred: the package itself is stdlib only

        self._client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model
        self.max_tokens = max_tokens

    def __call__(self, system: str, user: str, schema: dict) -> dict:
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
            output_config={"format": {"type": "json_schema", "schema": schema}},
        )
        return parse_json_message(msg)


def parse_json_message(msg) -> dict:
    stop = getattr(msg, "stop_reason", None)
    if stop == "refusal":
        raise ModelUnusable("model refused")
    if stop == "max_tokens":
        raise ModelUnusable("response truncated at max_tokens; raise the ceiling")
    # Join every text block: a response may start with a thinking block, and
    # text can legitimately arrive in more than one block.
    text = "".join(getattr(b, "text", "") for b in msg.content
                   if getattr(b, "type", None) == "text")
    if not text.strip():
        raise ModelUnusable("no text block in the response")
    return json.loads(text)
