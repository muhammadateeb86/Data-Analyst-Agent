# Churn Data Analyst — Autonomous Data Analyst

An agent that answers natural-language questions about a telco churn dataset
by planning tool calls (EDA queries, a trained churn model, and an
aggregate-risk tool), executing them for real, verifying the results, and
synthesizing an answer where every number is traceable to a tool output —
never something the LLM wrote itself.

## What's here

```
notebook/churn_eda_and_training.ipynb   EDA + training + evaluation (Stage 1)
src/model/                              cleaning.py, train.py, predict.py — the model as a callable
src/tools/                              data_tools.py (sandboxed EDA), model_tool.py, segment_tool.py
src/agent/                              planner → executor → verifier → synthesizer loop (Stage 3)
src/app/streamlit_app.py                chat UI, wired live to the agent (Stage 2)
tests/                                  pytest suite for each layer
data/                                   raw + cleaned CSV, fitted pipeline, metrics.json
docker/Dockerfile                       containerized app
```

## Running it

```bash
pip install -r requirements.txt
cp .env.example .env        # fill in GROQ_API_KEY or OPENROUTER_API_KEY
python -m src.model.train   # only needed if you want to retrain; data/model_pipeline.joblib is already committed
streamlit run src/app/streamlit_app.py
```

Or with Docker:

```bash
docker build -f docker/Dockerfile -t churn-analyst .
docker run -p 8501:8501 --env-file .env churn-analyst
```

Tests: `pytest -q` (82 tests, no API key required — the agent tests use an
injected fake LLM).

## Data issues found and how they were handled

Full detail and the investigation behind each decision is in the notebook
and in `src/model/cleaning.py`'s docstring; summary:

- **`TotalCharges` is a string column with 11 blank values.** All 11 belong
  to `tenure == 0` customers (brand-new accounts that haven't been billed
  yet), and `TotalCharges ≈ MonthlyCharges × tenure` for every other row
  (r ≈ 0.9996). The blanks are filled with `0.0` — the structurally correct
  value for zero tenure — rather than a mean imputation, which would
  fabricate billing history that doesn't exist. The code re-checks this
  `tenure == 0` assumption every run and raises instead of silently
  imputing if a future data refresh breaks it.
- **73 rows share an identical feature profile** with at least one other
  row (all at `tenure == 1`, where `TotalCharges == MonthlyCharges` removes
  a degree of freedom and pricing is effectively quantized). These are
  genuinely different customers who happen to look alike in their first
  month, not duplicates — kept as-is. 18 of the 33 groups have **mixed
  churn outcomes** for the identical profile, which sets a hard ceiling on
  achievable accuracy for those rows: no classifier can separate them on
  the given features. This is a real, not fixable, part of the gap between
  the model's ROC-AUC and a hypothetical perfect score.
- No duplicate `customerID`s, no malformed IDs, no stray whitespace in
  categorical columns, and the `"No phone service"` / `"No internet
  service"` labels are consistent with `PhoneService` / `InternetService`
  (not data errors).

## Model and metric choice

Two candidates were trained and compared: logistic regression and a random
forest, both with `class_weight="balanced"` (churn is the minority class,
~26.5%). Logistic regression was shipped:

| | ROC-AUC | PR-AUC |
|---|---|---|
| Logistic regression | 0.842 | 0.633 |
| Random forest | 0.842 | 0.651 |

The two models are within ~0.0005 ROC-AUC and ~0.018 PR-AUC of each other —
not a meaningful difference given the mixed-outcome ceiling described
above. **ROC-AUC was the primary metric, with PR-AUC reported as a
secondary check** — not plain accuracy. `risk_score` is explicitly a
continuous ranking output, not a label, and the assignment's own use cases
("which customers are most likely to churn," "aggregate churn risk across
segments") are ranking/scoring tasks; ROC-AUC measures ranking quality (the
probability a random churner scores higher than a random non-churner)
independent of any decision threshold, which fits that directly. Accuracy
is rejected outright: with ~26.5% positive class, a trivial always-predict-
"No" model scores ~73.5% accuracy while being useless. ROC-AUC alone can
look optimistic under imbalance, so PR-AUC (average precision) is reported
alongside it as an honest cross-check — it's more sensitive to performance
on the minority (churn) class, which is the class a retention team actually
acts on. See `notebook/`, Section 4, for the full walkthrough.

Given the near-tie, the decision came down to what the rest of the system
needed: logistic regression gives an exact, per-prediction signed
coefficient explanation (`top_factors` in every prediction — genuinely
computed as `contribution_i = encoded_value_i × coef_i`, not a fixed global
importance list) with no extra explainability library, and is far cheaper
to call repeatedly — which matters directly for `model/segment_risk`,
which scores every row in a segment in one vectorized call.

## How the agent plans, executes, and verifies

`DataAnalystAgent.answer()` is a small, bounded loop — **plan → execute →
verify → synthesize**, with up to two re-plans (three total planning
attempts) before it gives up and raises instead of looping forever:

1. **Plan** (`planner.py`) — one LLM call returns JSON steps against a
   fixed, small tool surface: `data/count_rows`, `data/query` (an
   AST-sandboxed pandas expression — no imports, no builtins, no
   mutation), `data/describe`, `data/filter`, `data/groupby_aggregate`,
   `model/predict` (real or hypothetical), `model/batch_predict`,
   `model/project` (current-vs-projected risk for one customer, tracing
   back to a verified feature snapshot), and `model/segment_risk`
   (aggregate risk across a segment — see below). The prompt includes the
   dataset's **actual schema** (column names, dtypes, and the real category
   values for low-cardinality columns), generated fresh from the live
   dataframe — not hardcoded — so the planner can't invent plausible-
   sounding column names that don't exist.
2. **Execute** (`executor.py`) — every step runs against the real
   dataframe/model; a failed step is recorded as an `{"error": ...}` fact
   rather than raising, so it becomes evidence for the next stage instead
   of silently vanishing. `Planner._normalize_step` also absorbs a couple
   of malformed-but-recoverable JSON shapes free-tier models tend to emit
   (see the "bug found and fixed" log below) instead of failing the whole
   plan over formatting.
3. **Verify** (`verifier.py`) — deterministic, not another LLM call: any
   fact with an `error` key or an empty (`row_count == 0`) result fails
   verification. On failure, the agent re-plans with the specific failure
   reason appended to the prompt.
4. **Synthesize** (`synthesizer.py` + `grounding.py`) — the LLM drafts
   prose that may only reference facts as `{{fact_N}}` placeholders; it is
   explicitly told never to write a digit itself. `ground()` then
   mechanically rejects the draft if any other numeral slipped in, and
   substitutes each placeholder with the actual tool value (rendered as
   readable text for dataframe/series results, compact JSON for scalars
   and small prediction dicts) — so a hallucinated number is a hard error,
   not a prompt-following convention. If the LLM's draft still fails this
   check, a deterministic fallback template ("Verified tool result: ...")
   is used instead of discarding the (already-correct) computed facts.

**Retry budget: why 2, and why it's not the whole story.** Early testing
used `retry_limit=1` (one re-plan, two total attempts). In practice a
single malformed tool call — e.g. the LLM emitting `data/query` with empty
`arguments: {}`, which fails with a plain Python `TypeError: query()
missing 1 required positional argument: 'expression'` — sometimes needed a
second re-plan to recover, and the failure only became visible as "resend
the same question and it works," which means the retry mechanism itself
was working correctly, it just didn't have enough budget. `retry_limit` is
now 2 (three total planning attempts) by default. Separately, a `429 Too
Many Requests` from the LLM provider is *not* a planning problem — it's a
transport-level rate limit — so it's retried at that layer instead:
`ChatCompletionProvider._send_with_rate_limit_backoff` (`providers.py`)
retries a 429 specifically, up to twice, honoring the `Retry-After` header
when present (falling back to exponential backoff otherwise), without
spending any of the agent's plan-retry budget on it. Any other HTTP status
is treated as a real failure and is not retried — blindly retrying a 400
or 401 wouldn't help and would just hide a real bug.

**Aggregate risk across segments** ("which Contract type has the highest
average predicted risk?") is the one case that doesn't fit a single-shot
plan-then-execute step cleanly — it needs the model applied to *every*
matching row and then grouped, which a one-shot planner can't do by first
fetching IDs and only then deciding what to predict. Rather than adding
multi-turn re-planning (which would blow the free-tier rate-limit budget),
this is a single composite tool, `model/segment_risk {by, filters?}`
(`src/tools/segment_tool.py`), that filters, scores the whole slice with
one vectorized `pipeline.predict_proba` call, and groups — deterministic
and cheap, in exactly one planned step.

**Never invent a number**, concretely: the sandboxed `data/query` AST
validator only allows a fixed method whitelist (no `eval`, no imports, no
`__`-prefixed access); the synthesizer's placeholder-only contract plus
`ground()`'s numeral rejection makes it structurally impossible for prose
to contain a number that didn't come from a fact; and `model/segment_risk`
means aggregate-risk answers are computed in one deterministic call instead
of the LLM being asked to average anything itself. The AST sandbox is
covered by 33 adversarial cases in `tests/test_data_tools.py` — dunder-chain
traversal (`().__class__.__bases__[0].__subclasses__()`), `getattr`/
`globals`/`vars` indirection, bare `eval`/`exec`/`open`/`compile`,
comprehensions and lambdas, mutating or file-writing methods (`to_csv`,
`assign`, `replace`, `drop`, `pop`, `insert`, `df.eval`, `df.query`),
assignment/import statements, starred-argument unpacking, and the walrus
operator — all rejected, alongside a positive test confirming a realistic
multi-step chain (filter → groupby → multi-aggregate → sort → head) still
works.

## Bugs found and fixed during review

**`ground()` false-positive on double-digit fact IDs.** `_NUMERAL`'s
negative lookbehind only blocks a match starting *immediately* after a
letter/underscore. For a placeholder like `{{fact_10}}`, the lookbehind
correctly blocks a match starting at the `1` (preceded by `_`), but the
regex engine then tries the next position and matches the trailing `0` on
its own — preceded by `1`, which isn't a letter or underscore — so
`ground()` rejected a perfectly valid placeholder as a "hallucinated
numeral." This broke synthesis outright for any ledger with 10 or more
facts (a wide `describe()`, a `groupby` over several categories, several
batch predictions) — an uncaught `ValueError` that Streamlit's outer
`except` turned into an opaque "Unable to answer this request" for a class
of legitimate multi-step queries. **Fix:** strip well-formed `{{fact_N}}`
placeholders out of the template *before* scanning for stray numerals,
instead of scanning the raw template. This keeps the original guarantee
exactly (any digit outside a valid placeholder is still rejected — verified
with a regression test asserting `"Value 99 and {{fact_11}}"` still raises)
while removing the false positive on multi-digit IDs. See
`src/agent/grounding.py::ground()` and
`test_grounding_accepts_double_digit_fact_ids` in `tests/test_agent_core.py`.

**Missing `from typing import Any`** in `grounding.py` — three functions
annotated parameters as `Any` without importing it. This didn't break
normal calls (`from __future__ import annotations` defers annotation
evaluation to strings), only anything that introspects the function's type
hints (`typing.get_type_hints`, some linting/IDE tooling) — exactly the
kind of thing that looks fine in a manual test and breaks unpredictably
later. Fixed.

**Dead code removed:** `execute_dataframe_query` (`data_tools.py`) and the
`predict_churn_risk_batch` alias (`model_tool.py`) were defined and
exported but never called by the agent, executor, app, or any test.
Removed, along with a stale comment in `executor.py` claiming `query` and
`customer_features` were "internal, not exposed in normal planning" —
false; both are explicitly documented and steered toward in the planner
prompt.

**Inconsistent exception contract:** `answer()` raised a bare `PlanError`
(a `ValueError`) when retries were exhausted during *planning*, but a
`RuntimeError` when retries were exhausted during *verification* — two
different exception types for the same "gave up" outcome. Unified both
paths to raise `RuntimeError`.

**Plan-shape normalization gap:** the planner's own prompt describes tools
using `tool/action` slash notation (e.g. `model/project`), and free-tier
models sometimes echo that notation straight into the JSON `"tool"` field
— `{"tool": "model/project", "arguments": {...}}` — instead of splitting it
into separate `"tool"`/`"action"` keys. This was previously rejected
outright as `"Plan contains an unknown tool"`. `Planner._normalize_step`
now handles this shape (and the `{"data/describe": {...}}` compact form)
before validating.

## Known limitations / what I'd do differently with more time

- The verifier is all-or-nothing per plan: if one step in a multi-step plan
  errors, the whole answer fails and retries, even if the other steps
  produced a perfectly good answer to a *different part* of the question.
  A partial-credit verifier (answer what's grounded, note what wasn't)
  would be a better user experience.
- Free-tier LLM planning quality is the biggest variable in practice — the
  schema injection and explicit guidance in the prompt (see `planner.py`)
  for percentage questions (compute `*100` directly, so the fact is already
  in percent — an earlier version left this to the LLM's phrasing and it
  mislabeled a raw 0–1 fraction as "0.42 percent"), "which category has the
  highest rate of X" questions (a single grouped-boolean `data/query`, not
  `idxmax` alone, since the rate itself is also needed), and named-customer
  hypotheticals (route through `model/project`, never a partial `{features}`
  dict to `model/predict` directly, which requires every column and fails)
  meaningfully reduce hallucinated columns and bad tool choices, but can't
  fully eliminate them; the bounded retry budget (now 2 re-plans) is the
  safety net for what slips through.
- `data/filter` returns up to 100 raw rows for the user to read — fine for
  "show me some examples," but a plan that reaches for `filter` on a
  "what percentage" question produces a wall of rows instead of a number.
  The prompt steers percentage/rate questions toward a `data/query`
  boolean-mean expression instead, but this is prompt guidance, not a hard
  constraint.
- The planner's `json.loads()` call has no markdown-fence stripping. Some
  free-tier models wrap JSON output in ` ```json ` fences despite being
  told not to, which would burn a re-plan attempt on a formatting issue
  rather than a real planning failure. Not yet hit in testing, but not
  guarded against either.
