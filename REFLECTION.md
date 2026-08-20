# Reflection

## Hardest Part

The hardest part was making the agent reliably answer multi-step questions involving both the dataframe and the churn model. A question such as "which Contract type has the highest average predicted churn risk?" requires more than retrieving existing data: the system must identify the relevant customers, generate predictions for them, and then aggregate those predictions. I initially tried to have the LLM orchestrate these dependent steps, but this was unreliable on a free-tier model. I therefore introduced a deterministic `model/segment_risk` tool that performs the filtering, vectorized prediction, and aggregation in Python. This kept the reasoning with the agent while keeping numerical computation outside the LLM.

## What I Learned

The most important lesson was that prompting alone is not enough to guarantee numerical correctness. I initially relied on the LLM to follow instructions not to invent numbers, but testing showed that it could still produce numerical literals. I changed the design so the synthesizer can only reference verified facts using placeholders, which are replaced by actual tool results deterministically. I also learned that safety mechanisms need adversarial testing: testing only normal inputs gave false confidence, so I added tests for attempts to bypass the dataframe sandbox and found and fixed a grounding bug involving double-digit fact IDs.

I also learned the importance of designing around real constraints. Free-tier rate limits made unlimited retries impractical, so I used bounded planning retries and separate rate-limit backoff at the provider layer.

## What I Would Improve

With more time, I would add a small evaluation set to measure planning accuracy and numerical grounding systematically rather than relying mainly on manual testing. I would also make verification support partial results instead of discarding an entire multi-step answer when one step fails.

## Time and AI Use

I spent approximately 7–8 hours. A significant portion of the time was spent testing, debugging, and hardening the system rather than writing the initial code.

I used AI to help understand the assignment, this really helped me alot, and then offcourse I don't remember every syntax so i useed it to write parts of code. 