# Reflection

**Hardest part.** Getting the agent to *aggregate* the model's predictions
rather than just look things up. Per-customer lookups (`model/predict`,
`model/project`) and dataframe EDA (`data/query`, `groupby_aggregate`) are
each straightforward on their own, but "which Contract type has the highest
average *predicted* churn risk" needs both at once — score every matching
row with the model, then group the scores — and the agent only gets one
planning call with no re-planning mid-execution. The fix was to stop trying
to make the LLM orchestrate that in two dependent steps and instead give it
one deterministic composite tool (`model/segment_risk`) that does the
filter → vectorized score → groupby itself, server-side, in a single call.

**What I learned / had to teach myself.** Getting reliable "never invent a
number" behavior out of a free-tier model isn't really a prompting problem
you can fully solve with instructions — the model will occasionally still
write a number in its draft. The reliable fix was mechanical: force the
synthesis step to only ever reference facts through `{{fact_N}}`
placeholders, then reject the draft outright with a regex if any other
digit shows up, and fall back to a deterministic template built straight
from the fact ledger. That turns "please don't hallucinate" from a hope
into something the code actually enforces — but it also taught me that a
mechanical guarantee is only as good as the regex behind it: a later review
caught a real bug in exactly this mechanism (the numeral-rejection regex
false-positived on any fact ID of 10 or higher — `{{fact_10}}` — because
its lookbehind only blocks a match starting *immediately* after a letter or
underscore, not every digit in a longer run; `fact_10`'s trailing `0` is
preceded by `1`, not `_`, so it still matched). The fix was small, but
finding it required deliberately testing the ledger past nine facts and
re-reading the regex character by character rather than trusting that it
"looked right" — the same lesson as the AST sandbox, really: a safety
mechanism you haven't adversarially tested is a claim, not a guarantee. I
went back and built out real adversarial coverage for the `data/query`
sandbox itself (dunder-chain traversal, `getattr`/`globals`/`vars`
indirection, comprehensions/lambdas, mutating/file-writing methods,
assignment/import statements, starred-argument unpacking, the walrus
operator — 33 cases total) — the original two test cases passed, but they
didn't prove much on their own.

**A separate lesson from live usage, not a code review.** After the app
was working, I still occasionally hit "wrong tool called" or a raw
Python error surfacing to the chat — and then resending the *exact same
question* would work. That's a real signal: it means the self-check/retry
mechanism was doing its job, it just didn't have enough budget. One
malformed single-step plan (e.g. the LLM emitting `data/query` with no
`expression` argument) sometimes needed a second re-plan, not just one, to
recover — so I bumped `retry_limit` from 1 to 2 (three total planning
attempts). Separately, a `429 Too Many Requests` from the LLM provider
turned out not to be going through the agent's retry loop at all — it's a
transport-level error raised below the planner, so the agent's re-plan
budget never even saw it, and it failed on the first try every time. That
one needed a fix in a different place: a short, bounded backoff-retry
inside the provider itself (honoring `Retry-After` when the API sends one),
not more agent-level re-planning — retrying a rate limit by re-planning
faster would just hit the rate limit harder, which the brief specifically
warns against ("designing an efficient loop is part of the challenge, not
a bug to route around with a paid tier").

**What I'd do differently with more time.** Make the verifier partial-
credit instead of all-or-nothing — right now one failed step in a
multi-step plan discards every other (perfectly good) fact and forces a
full re-plan. I'd also add a small eval set to get an actual hallucination-
rate number instead of relying on spot-checking transcripts by hand, and
guard the planner's `json.loads()` against markdown-fenced JSON output,
which some free-tier models produce despite being told not to.

**Time.** I roughly spend about 7 to 8 hours, I think. Writing code is not a big deal but 
making it work correctly it the the thing. There were bugs and then the real testing cost 
Alot of time. 

**AI tool use.** I used the Ai too for understanding the assignment. It helped me alot to understand what is required, then offcourse i dont remmeber all the syntax so i used it for writing parts of code too. 