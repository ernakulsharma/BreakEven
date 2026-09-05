# Open Risk Engine

Transaction fraud prevention and chargeback representment, in three pipelines built on one dataset.

The problem with conventional risk tooling is that it optimises the wrong thing. Rigid rules and off-the-shelf scoring APIs are tuned to catch fraud, so they measure themselves on recall — and a system optimised for recall will happily decline a customer worth $3,000 to stop $40 of fraud. False declines cost merchants substantially more than fraud does. This engine treats the decline decision as an economic one, and treats the dispute that follows as a structured evidence problem rather than a manual one.

```
Phase 1  prevention   IEEE-CIS -> XGBoost -> per-transaction cost-optimal threshold
Phase 2  abuse rings  entity graph from IEEE-CIS's own linking columns
Phase 3  disputes     reason-code rules + synthetic case data + local LLM narrative
```

---

## Quick start

```bash
pip install -r requirements.txt

# generate a runnable fixture (no Kaggle account needed)
python -m tests.make_sample --rows 30000 --out data/sample

python -m phase1_prevention.train --data-dir data/sample   # train
python -m tests.smoke_test                                 # verify all 3 phases
python demo.py --data-dir data/sample                      # full walkthrough

uvicorn phase1_prevention.app:app --port 8000              # serve
```

**The sample fixture is for proving the code runs, not for measuring the model** — its labels come from a synthetic logit and carry almost no learnable signal.

---

## Where the real dataset goes

```
data/
  ieee-cis/
    train_transaction.csv     590,540 rows, has isFraud    <- everything reads this
    train_identity.csv        144,233 rows, joins on TransactionID
  sample/                     generated fixture, safe to delete
```

`data/` is gitignored, so a 1.3 GB dataset never enters the repo. To keep it outside the checkout entirely — shared between clones, or on a different volume — set `RISK_DATA` and the default path follows it:

```bash
export RISK_DATA=/mnt/datasets/riskengine
# data-dir now defaults to /mnt/datasets/riskengine/ieee-cis
```

Check the directory before spending an hour training on it:

```bash
python -m tests.check_data --data-dir data/ieee-cis
```

Then all three phases read that one directory:

```bash
python -m phase1_prevention.train --data-dir data/ieee-cis   # phase 1, writes artifacts/
python -m tests.smoke_test        --data-dir data/ieee-cis   # all three phases
python demo.py                    --data-dir data/ieee-cis   # walkthrough
```

### Only two of Kaggle's four files are usable

The competition bundle also ships `test_transaction.csv` and `test_identity.csv`. **They carry no `isFraud` column** — Kaggle scored submissions against a hidden leaderboard and never released the ground truth. They cannot train, calibrate, or validate anything here, and "test set" is exactly what you reach for when you want to validate a model. `check_data` fails loudly if you point it at them.

### There is no separate validation file, by design

All three splits are cut *temporally out of `train_transaction.csv`* by the trainer. Fraud patterns drift, so a future-holdout is the only split that measures what deployment will feel; a random 80/20 leaks next month's behaviour into training and flatters every number. `check_data` previews the exact cut the trainer will make:

```
slice               rows     fraud      rate   role
fit               20,400       829    4.064%   booster training
calibrate          3,600       171    4.750%   early stopping + probability map
test               6,000       248    4.133%   never seen until the final report
```

Ratios live in `common/config.py` (`TEST_SPLIT_FRACTION`, `CALIBRATION_FRACTION`). Watch the calibration slice's positive count: below 200 it falls back from isotonic to Platt scaling, which `check_data` warns about in advance.

### Don't use `--limit` to speed up Phase 2

`--limit` on `demo.py` and `smoke_test.py` takes the *earliest* N rows, and rows are ordered by `TransactionDT`. Phase 2 links consecutive transactions per entity inside a 7-day window, so compressing the calendar span packs far more pairs into that window and the graph over-connects. On the fixture, the first 8,000 rows produce a largest component of **70.1%**; a *random* 8,000 rows produce **0.7%**. Same row count. Both commands warn when `--limit` is set, and the smoke test skips its giant-component assertion rather than failing for a reason that is about the slice. Phases 1 and 3 are unaffected.

Nothing is a hard dependency. Redis falls back to an in-process store, Ollama falls back to a deterministic letter template, and an untrained model falls back to a transparent heuristic. Every fallback reports itself in its output rather than pretending to be the real thing.

---

## Phase 1 — prevention

A single well-calibrated XGBoost model, with all cost-sensitivity pushed into the decision threshold.

**The threshold is computed per transaction.** Expected cost of approving is `p x C_FN`; expected cost of declining is `(1-p) x C_FP`. Decline when the first exceeds the second, which gives `p > C_FP / (C_FP + C_FN)`. Because `C_FP` includes the customer's lifetime value, a valuable customer is automatically held to a higher standard of evidence before being declined:

| Transaction | Cost if fraud is missed | Cost if wrongly declined | Decline threshold |
|---|---|---|---|
| $5 | $30 | $42 | 0.583 |
| $80 | $105 | $61 | 0.366 |
| $800 | $825 | $241 | 0.226 |
| $3,000 | $3,025 | $791 | 0.207 |

### The score has to be a probability

This is the part that took the longest to get right, and it was wrong for most of the build.

`p > C_FP / (C_FP + C_FN)` compares a model output against a threshold derived from dollars. That comparison only means anything if `p` is a posterior probability — the number that, across all transactions scoring 0.30, means 30 of every 100 are fraud. A booster trained with `scale_pos_weight=23` does not produce that number. Reweighting the positive class to fight a 3.5% base rate is good for optimisation and harmless for ranking, but it shifts the output scale: the model is fitting a reweighted world where fraud is roughly half the data.

Measured on the fixture before `common/calibration.py` existed:

| | raw | calibrated |
|---|---|---|
| mean predicted P(fraud) | 0.2885 | 0.0472 |
| predicted / actual | **6.98x** | 1.14x |
| Brier score | 0.1055 | 0.0377 |
| expected calibration error | 0.2471 | 0.0108 |

Raw Brier was *worse than always predicting the base rate* (0.0396). Ranking was fine; the probabilities were fiction, and every dollar figure computed from them was fiction too.

The fix is a monotone map fitted on a held-out temporal slice — isotonic where there are enough positives to support it, Platt scaling where there are not. Because it is monotone, PR-AUC and ROC-AUC are unchanged by construction and the recall-targeted baseline is unaffected (it picks a quantile). Only the cost-aware policy moves, which is exactly right: it is the only one that reads `p` as a number rather than as an ordering.

This forced a third temporal slice. The calibration map cannot be fitted on data the booster trained on, and — separately — early stopping was previously selecting `best_iteration` against the *test* set, tuning the model on the benchmark it was then reported on. Both now use the calibration slice, and the test window is untouched by anything until final evaluation.

```
fit (68%) ────────────► calibrate (12%) ────────────► test (20%)
booster                 early stopping +              never seen until
                        isotonic/Platt fit            the final report
```

### Results on real data

| Policy | Recall | FPR | Total $ cost |
|---|---|---|---|
| Fixed global threshold (cost-blind) | 90.0% | 29.1% | $6.85M |
| Per-transaction Bayes threshold | 62.8% | 8.9% | $2.21M |

PR-AUC 0.494, ROC-AUC 0.905. The cost-aware policy catches *less* fraud in raw recall terms and that is the point: it declines to chase low-value fraud when doing so means declining expensive customers.

> **These figures predate calibration and have not been re-run.** They were produced by the two-way split with uncalibrated scores, so the ranking metrics (PR-AUC, ROC-AUC) still stand — calibration is monotone — but the recall, FPR and dollar columns for the Bayes row do not. Re-run `python -m phase1_prevention.train --data-dir data/ieee-cis` before quoting them anywhere. Expect the cost reduction to fall, and read the `[diag]` lines before believing whatever replaces it.

End-to-end online latency — feature store read, feature construction, inference, and policy — measures **p50 0.66ms / p95 0.87ms**, well inside the sub-100ms budget.

### Three things that did not work

**Dollar costs as gradient weights.** The obvious way to build a cost-sensitive model is to weight each row by its dollar impact. It collapsed PR-AUC from 0.494 to 0.08 and never converged: a few high-cost rows dominated every boosting round, and the model stopped learning general splits. The dollar saving was 2.6%. The function is kept in `decision.py` history as a documented dead end — separating the probability estimate from the decision policy is the correct decomposition.

**Unbounded LTV.** The first cost model used cumulative card spend as the LTV term. `card1` in IEEE-CIS is a hashed bucket shared by ~44 transactions, not a customer id, so that figure reached ~$1.6M and made false-positive cost ~1,500x false-negative cost. The policy degenerated to declining almost nobody (8.9% recall). It is now bounded: typical basket size x 12 expected future purchases, hard-capped.

**Uncalibrated probabilities.** Covered above. Worth listing as a failure rather than an improvement, because the system did not error — it produced plausible dollar figures from a score that was 7x too high, and two rounds of debugging went into the *cost model* before anyone checked whether `p` was a probability.

The trainer emits a diagnostic that separates three failure modes, because low recall means completely different things depending on cause:

```
[diag] median FP/FN cost ratio 1.79x | median Bayes threshold 0.642 | 0.02% of scores clear it
[diag] calibrated scores reach p99=0.157, max=0.711
```

- **Ratio above ~20x** — the cost model is inflated. This is the unbounded-LTV failure. Fix the cost model.
- **Sane ratio, threshold reachable, low recall** — the model is too weak to clear the bar. Fix the model. The cost saving is real but uninteresting: it comes from declining nothing.
- **Sane ratio, threshold *unreachable*** — the third case, and invisible until calibration landed. The cost model asks for `p > 0.64` while 99% of calibrated scores sit below 0.16. No calibrated model on a 3.5% base rate emits that confidence often, so *no amount of model improvement fixes this*. Either the decision is not really binary — route the band to review rather than forcing approve/decline — or `CHURN_PROB` / `LTV_CAP` overstate what one decline costs.

That last case is the one uncalibrated scores were hiding: sitting 7x too high, they cleared a 0.64 threshold by accident, which looked like a working policy.

### The review band is measured, not assumed

`decide()` serves three actions but the trainer only ever measured two, so the policy being reported was not the policy being served — the same train/serve skew `common/features.py` exists to prevent, one level up in the stack. The trainer now reports the band:

```
[review] band routes 0.08% of traffic; it holds 1.6% of all fraud at 80.0% precision.
```

No dollar cost is assigned to it. An analyst hour is a real number the operator has and this repo does not, so putting one in the cost column would make that column fiction.

---

## Phase 2 — abuse rings

Built from IEEE-CIS's own linking columns rather than a separate review-fraud dataset, so ring detection runs on the same transactions the model scores. Nodes are transactions; two are linked when they share a card fingerprint, device or uid.

Three constraints make the difference between a useful graph and a blob:

**1. Fields must be identifying.** `P_emaildomain` is only the *domain*, so "gmail.com" links ~40% of the dataset. The graph now measures average degree per field and drops any field behaving like a category rather than an entity. On the fixture this automatically rejects email at 3,330 average degree.

This also corrects the original plan, which listed `addr1`/`addr2` as ring-linking fields. They are not street addresses — IEEE-CIS `addr1` has roughly 332 distinct values across 590,540 rows (average degree ~1,780). It is a coarse billing-region code and cannot identify a drop address.

**2. Hub limits must scale with the dataset.** A fixed limit of 500 is strict on 30k rows and useless on 590k. Limits are `max(floor, rate x n_rows)`.

**3. Rings are bursts.** This is the constraint that mattered most. Linking two transactions four months apart through a shared entity is coincidence, not collusion. Each entity's transactions are sorted by time and only *consecutive* pairs inside a 7-day window are linked, so long gaps break the chain with no bucket-boundary artifacts:

| Link window | Largest component | Rings | Ring recall | Precision |
|---|---|---|---|---|
| none (star topology) | 32.5% | 3 | 30.5% | 100% |
| 30 days | 30.0% | 8 | 45.3% | 100% |
| **7 days (default)** | **0.2%** | **11** | **51.6%** | **100%** |
| 1 day | 0.1% | 12 | 51.1% | 100% |

Without it, 99% of transactions collapsed into a single component that then got *scored as a ring* ("29,728 cards on 354 devices"). Components above 1% of traffic are now rejected as construction artifacts and the fact is surfaced, not hidden.

Ring scoring is a transparent weighted blend of five named behaviours — card fan-out per device, address concentration, velocity, amount uniformity, and new-account share — not a second black box. Output goes to an analyst who has to act on it, so every point of the score is attributable. Flagged rings run at 24x the base fraud rate on the fixture.

Uses `scipy.sparse.csgraph` rather than networkx: 590k transactions is well past where a Python-object graph fits in memory.

---

## Phase 3 — chargeback representment

**CUAD was dropped from the plan.** It is 510 commercial legal contracts annotated for clause extraction — indemnification, governing law, IP assignment. Nothing in it resembles a transaction record, delivery log, 3DS result, or reason code, so it cannot teach a model anything about representment. And no public dataset combines transaction, delivery, authentication and dispute-outcome data, because that data lives inside proprietary merchant and acquirer systems.

So the input side is generated and the output side is rule-driven:

- **`reason_codes.py`** encodes nine Visa and Mastercard codes with the evidence each requires, weights, response deadlines, and digital-goods substitutions (a SaaS charge has no carrier, so delivery requirements become access logs — applying the physical template guarantees failure on evidence that never existed).
- **`synth.py`** generates cases across a deliberate quality spread — strong, partial and weak — because an engine that only sees winnable cases learns nothing. It can seed from real IEEE-CIS rows so amounts and identifiers stay consistent across all three phases.
- **`evidence.py`** resolves each required item against the case, scores a win probability, and recommends represent / partial / accept liability. A missing high-weight *required* item is not a gradual penalty — no proof of delivery on a "not received" claim loses the case regardless of peripheral evidence.
- **`packet.py`** ranks the queue by expected recovery (`win probability x amount`), since representment costs staff time per case.

**The LLM's role is deliberately narrow.** Rules decide what evidence is required, what is present, whether to fight, and the win probability. The model only writes the rebuttal letter from that already-resolved structure. It never determines an outcome and never sees a question it could answer wrongly in a way that costs money — every factual claim the letter can make already exists as a resolved evidence item, so hallucination risk is bounded by construction. If Ollama is unavailable the template runs and the packet is still submittable; the narrative source is recorded either way.

The win-probability priors are **configuration**, seeded from published industry representment rates. Replace them with your own outcome data after fifty settled disputes. Until then they are an ordering signal, not a forecast to quote to anyone.

---

## Layout

```
common/
  config.py       cost model, hub limits, thresholds — every money-affecting tunable
  schema.py       Pydantic contracts for all three phases
  features.py     feature engineering shared by trainer and API
  calibration.py  isotonic/Platt map + reliability measurement
phase1_prevention/
  train.py        memory-safe IEEE-CIS ingest, 3-way split, training, calibration
  decision.py     Bayes-optimal threshold, cost accounting, capacity guardrail
  feature_store.py  Redis with in-memory fallback
  prevention.py   the online scoring path
  app.py          FastAPI
phase2_rings/
  build_graph.py  entity graph, hub pruning, identifiability guard, time windows
  rings.py        ring scoring and evaluation against labels
phase3_disputes/
  reason_codes.py Visa/Mastercard rulebook
  evidence.py     evidence resolution and win scoring
  synth.py        case generator
  packet.py       packet assembly, Ollama, triage
tests/
  check_data.py   preflight: labels, columns, split preview, field identifiability
  make_sample.py  IEEE-CIS-shaped fixture with planted rings
  smoke_test.py   30 checks across all three phases, calibration, and the API
demo.py           end-to-end walkthrough
```

`common/features.py` exists specifically to prevent train/serve skew. The batch path derives entity aggregates from a leak-free cumulative groupby; the online path reads them from Redis; both then call the *same* arithmetic. Offline aggregates use `cumsum`/`cumcount` rather than `expanding().mean()`, which was the original cause of OOM kills on a 1-core/3.9GB box.

---

## Known limits

- **The headline Phase 1 numbers above need re-running post-calibration.** See the note under the results table.
- Calibration is fitted once, offline. Drift is real and nothing here monitors it; the map should be refitted on the same cadence as the model, and `raw_model_score` is exposed on every response so the two scales can be compared in production.
- Win-probability priors are industry estimates, not fitted to your outcomes.
- Ring recall on the fixture is ~52%; small planted rings that share only a hub-pruned field stay invisible. Measure against real IEEE-CIS before trusting a number.
- The `uid` key (`card1|addr1|D1`) is an approximation of an account, not an account.
- Phase 2's graph is sensitive to the calendar span of its input, not just the row count. Subsampling by rows is not a valid speedup for it; subsample by time window or not at all.
- Ring recall has no ground truth on real IEEE-CIS — there is no ring label in the dataset. The smoke test skips that check and the fraud-lift check carries the weight instead.
- The cost model assumes a single gross margin and churn probability across all customers. Both are per-segment in reality, and both live in `common/config.py` for that reason.
- Reason-code requirements and deadlines change. Treat `reason_codes.py` as configuration to review against your acquirer's current guide, not as legal advice.
