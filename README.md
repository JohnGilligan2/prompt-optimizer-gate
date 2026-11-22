# prompt-optimizer-gate

A prompt optimizer that can propose and test prompt changes but can't ship them, and the regression gate that decides what it's allowed to stage.

## The problem

I run an AI assistant inside an MSP's helpdesk. It reads a new ticket, pulls similar past tickets and writes an internal note for the engineer, plus a draft customer reply. An LLM judge scores every output 1 to 5 with a one-line reason. After a few weeks of those critiques it's tempting to have a model read them and rewrite the prompt.

A model will write you a new prompt in one call. Deciding whether to trust it is harder, because the obvious gate gets fooled. The judge is noisy. One fixture flipped between a 2 and a 4 across two runs of the same prompt and produced a win that wasn't real. An average hides damage, so a candidate can lift the mean while one kind of ticket gets clearly worse. And replaying old tickets against today's helpdesk leaks the answer, because the ticket's own later messages, its final stage and next month's repeat of the same fault are all there for retrieval to find.

## What it's attached to

The assistant runs on the main support queue. Comparing June 2025 through February 2026 with April through August 2026, tickets closed within 72 hours went from 33 to 39 percent. The golden set behind the gate is 3,200+ cases mined from operational history, each one a model output set beside what the engineer actually did on that ticket.

## How the gate decides

A candidate prompt is replayed head to head against the current prompt on frozen fixtures, every fixture twice per arm (`GATE_REPEATS = 2`). It passes only if all of these hold:

- every candidate output passes the deterministic checks (dashes, smart quotes, AI tells, banned terms, a greeting that picked up a number)
- its mean judge score is at least the current prompt's
- no single fixture's mean drops by more than 1.0 judge point (`PARETO_TOLERANCE = 1.0`)
- every output still carries a required literal; in production that was the header a poller deduplicated on, so dropping it would have re-posted on every ticket

Passing isn't enough to stage. The candidate has to beat the current mean outright, and a tie stays on disk. A winner is saved as a new prompt version with no label. The generator only ever reads the version labeled `production`, so nothing changes until a person moves that label. Moving it back is the rollback, with no deploy either way.

Why 1.0? With two repeats, a single 2-to-4 flip on one repeat moves that fixture's mean by exactly 1.0, so a drop has to exceed it to count as real. If you want a tighter guard, raise the repeats before you lower the tolerance.

The optimizer side is two model calls a cycle, enforced by a hard cap in code. One reads the judge's critiques and names recurring weaknesses. The other rewrites only the editable parts of the prompt. Code puts the immutable middle back byte for byte and refuses a candidate that lost a required literal, deleted a rule family, shrank by half or grew past 1.6x. The gate's own replays never feed back into the corpus it learns from.

## Point-in-time replay

Each golden case has a cutoff, the moment before an engineer worked the ticket. Everything the generator sees goes through `Barrier`, a read-only wrapper over the helpdesk. It drops messages dated at or after the cutoff and hides the reference's evidence messages by id as well as by date. Search results lose the target and any ticket created after the cutoff. Stage and assignee are rewound from the change log. Writes are refused. Then `audit_inputs` checks the assembled inputs again and raises `LeakError` if anything from the future got through.

On the synthetic set in `example/`, a naive replay of the current prompt averages 2.83 and the point-in-time replay averages 2.50. The gap is one ticket whose fix the naive replay reads straight out of its own later messages.

## Run it

```
python -m pytest
python example/demo.py
```

The package is stdlib only (Python 3.10 or later); the tests need pytest. No API key. The generator, judge and optimizer model are deterministic stubs in `example/world.py`, so the whole loop runs offline in under a second. The demo checks the judge against human grades, builds the fixtures point in time, then runs two optimizer cycles. The first candidate raises the mean from 2.58 to 3.67 but drops one long ticket from 3.00 to 1.00, so the Pareto guard fails it and nothing is staged. The second wins everywhere and gets staged unlabeled. Production stays on version 1 until the demo's "human" promotes it.

## Using it on your own system

```python
from promptgate import PromptLayout, PromptRegistry, build_inputs, run_cycle
from promptgate.checks import make_checks

layout = PromptLayout(immutable_anchor="## CONTEXT", rules_anchor="## RULES",
                      required_literals=("[assistant-note]",))
fixtures = [{"id": c["case_id"], "reference": c["reference"],
             "inputs": build_inputs(my_helpdesk, c)} for c in golden_cases]

result = run_cycle(scored_cases, registry, layout, fixtures,
                   model=my_model,          # (system, user, schema) -> dict
                   generate=my_generator,   # (prompt_text, inputs) -> output
                   judge=my_judge,          # (fixture, output, repeat) -> {"score", "comment"}
                   checks=make_checks(banned_terms=["guaranteed"]),
                   required_literals=["[assistant-note]"])
# result.exit_code: 0 passed, 1 error, 2 failed the gate
# result.staged_version: an unlabeled version, or None
```

`promptgate/anthropic_model.py` is an optional `model` callable on the Anthropic API with schema-enforced JSON. It has not been run against the live API, and the tests do not cover it; the tests and the demo use the stubs in `example/world.py`. Your helpdesk only needs the five read methods listed at the top of `promptgate/pit.py`.

Calibrate the judge before you let it gate anything. That method is its own repo, [grade-the-grader](https://github.com/JohnGilligan2/grade-the-grader). `bands.judge_regression` and `kappa.py` here are the recurring check that the judge still lands where the humans did.

## Files

| file | what it does |
|---|---|
| `promptgate/gate.py` | the head-to-head gate, pass^k beside pass@k |
| `promptgate/pareto.py` | per-fixture guard, `GATE_REPEATS`, `PARETO_TOLERANCE` |
| `promptgate/pit.py` | the point-in-time barrier and the leak audit |
| `promptgate/optimizer.py` | one cycle: critiques, candidate, gate, stage |
| `promptgate/surgery.py` | editable segments only, validated before any gate spend |
| `promptgate/store.py` | versions and labels; staging is automatic, promotion is not |
| `promptgate/bands.py`, `kappa.py` | human bands and chance-corrected agreement for the judge |
| `promptgate/checks.py` | deterministic house-rule checks |
| `example/` | synthetic helpdesk, golden set, stubs, and the demo |

## License

MIT.
