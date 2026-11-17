# Stage 4: news calls stricter than the truth

Made by `src/review_stricter.py`. Each run's 64 news decisions are joined to the labelled test set
with pandas. A call is *stricter* when the pipeline's action is more severe than the true label
(keep < age-restrict < remove). Costs are the OpenRouter costs on each run's traced batches in LangSmith.

## Before and after

| | Before | After |
|---|---|---|
| Run finished (UTC) | 2026-11-15T19:44:49+00:00 | 2026-11-17T15:08:46+00:00 |
| News clips | 64 | 64 |
| **Stricter than truth** | **0** | **0** |
| Looser than truth | 13 | 15 |
| Match | 51 | 49 |
| Failed | 0 | 0 |
| Ana's clip (yt_c0143, truth age-restrict) | age-restrict (VC-03) | age-restrict (VC-03) |
| Total cost (7 / 7 batches) | $0.0293 | $0.0471 |
| Cost per clip | $0.000457 | $0.000736 |

## Review table: stricter calls

| Run | clip_id | Pipeline | Truth | Trace finding | Cause |
|---|---|---|---|---|---|
| before | none | | | No stricter calls, so no trace to open | No cause to fix |
| after | none | | | No stricter calls, so no trace to open | No cause to fix |

## Mismatches for reference (looser than truth)

| Run | clip_id | Pipeline | Truth |
|---|---|---|---|
| before | yt_c0033 | keep (VC-01) | age-restrict |
| before | yt_c0095 | keep (VC-01) | age-restrict |
| before | yt_c0177 | keep (VC-01) | age-restrict |
| before | yt_c0047 | keep (VC-01) | age-restrict |
| before | yt_c0144 | keep (VC-01) | age-restrict |
| before | yt_c0131 | keep (VC-01) | age-restrict |
| before | yt_c0178 | keep (VC-00) | age-restrict |
| before | yt_c0195 | keep (VC-01) | age-restrict |
| before | yt_c0072 | keep (VC-01) | age-restrict |
| before | yt_c0096 | keep (VC-01) | age-restrict |
| before | yt_c0196 | keep (VC-01) | age-restrict |
| before | yt_c0171 | keep (VC-01) | age-restrict |
| before | yt_c0120 | keep (VC-01) | age-restrict |
| after | yt_c0033 | keep (VC-01) | age-restrict |
| after | yt_c0095 | keep (VC-01) | age-restrict |
| after | yt_c0177 | keep (VC-01) | age-restrict |
| after | yt_c0047 | keep (VC-01) | age-restrict |
| after | yt_c0144 | keep (VC-01) | age-restrict |
| after | yt_c0131 | keep (VC-01) | age-restrict |
| after | yt_c0178 | keep (VC-01) | age-restrict |
| after | yt_c0060 | keep (VC-01) | age-restrict |
| after | yt_c0195 | keep (VC-01) | age-restrict |
| after | yt_c0072 | keep (VC-01) | age-restrict |
| after | yt_c0096 | keep (VC-01) | age-restrict |
| after | yt_c0113 | keep (VC-01) | age-restrict |
| after | yt_c0196 | keep (VC-01) | age-restrict |
| after | yt_c0171 | keep (VC-01) | age-restrict |
| after | yt_c0120 | keep (VC-01) | age-restrict |
