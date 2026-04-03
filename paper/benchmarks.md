# SMELT Benchmark Results

Generated: 2026-04-01 America/Los_Angeles

## Setup

- LM Studio base: `http://127.0.0.1:1234/v1`
- Live model used for inference: `Qwen 3.5 VL 122B A10B (8-bit, MLX)`
- Tokenizer: Qwen 3.5 VL 122B A10B tokenizer (local)
- Workspace: Production OpenClaw workspace (local)
- Audit date: `2026-04-01`
- Query source file: `USER.md` from production OpenClaw workspace
- Query-conditioned mode: `semantic`
- Query `max_records`: `8`
- Query `max_macros`: `12`

## Test 1: Latency (Time-to-First-Token)

Method: one warm-up run per format, then five measured runs each, alternating formats. TTFT is measured from HTTP request dispatch to the first streamed token returned by LM Studio. Because the live Qwen target can emit a leading reasoning-style token, TTFT is measured to the first streamed token of any kind.

- Effective startup files: `AGENTS.md, SOUL.md, USER.md, MEMORY.md`
- Today file exists: `False`
- Yesterday file exists: `False`
- Original bundle bytes: `27967`
- Original prompt tokens: `7268`
- SMELT semantic bundle bytes: `25191`
- SMELT semantic prompt tokens: `6715`

| Format | Mean TTFT (ms) | Std Dev (ms) |
| --- | ---: | ---: |
| Original markdown startup bundle | 14120.734 | 71.338 |
| SMELT semantic runtime startup bundle | 13272.695 | 147.825 |

### Raw TTFT Runs

| Format | Run | TTFT (ms) |
| --- | ---: | ---: |
| original_markdown | 1 | 14185.465 |
| smelt_semantic_runtime | 1 | 13288.856 |
| original_markdown | 2 | 14057.372 |
| smelt_semantic_runtime | 2 | 13093.077 |
| original_markdown | 3 | 14201.988 |
| smelt_semantic_runtime | 3 | 13500.620 |
| original_markdown | 4 | 14046.743 |
| smelt_semantic_runtime | 4 | 13219.760 |
| original_markdown | 5 | 14112.099 |
| smelt_semantic_runtime | 5 | 13261.164 |

## Test 2: Query Benchmark (10 Queries Through Query-Conditioned Retrieval)

| Query | Records Returned | Content Records | Output Bytes | Output Tokens | Byte Compression vs Source | Token Compression vs Source |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Who is Kane? | 4 | 1 | 112 | 38 | 97.73% | 97.16% |
| What is Edmund's health situation? | 5 | 2 | 270 | 89 | 94.52% | 93.35% |
| Tell me about Edmund | 15 | 8 | 954 | 293 | 80.63% | 78.10% |
| What does Edmund NOT like? | 15 | 8 | 954 | 293 | 80.63% | 78.10% |
| What are Edmund's hardware specs? | 11 | 5 | 414 | 136 | 91.59% | 89.84% |
| When was Edmund born? | 3 | 1 | 54 | 26 | 98.90% | 98.06% |
| What is Edmund's relationship with his mother? | 11 | 5 | 414 | 136 | 91.59% | 89.84% |
| What tools does Edmund use daily? | 13 | 6 | 456 | 151 | 90.74% | 88.71% |
| Summarize Edmund's family | 16 | 8 | 826 | 247 | 83.23% | 81.54% |
| What should I avoid when talking to Edmund? | 11 | 5 | 414 | 136 | 91.59% | 89.84% |

## Test 3: Baseline Comparison (Prompt Token Counts Per Query)

Method: token counts were measured locally with the tokenizer from the live LM Studio Qwen target, using the same prompt wrapper for all four approaches and changing only the context representation.

| Query | Raw Markdown | Heading-Stripped Markdown | Naive JSON | SMELT Semantic Runtime |
| --- | ---: | ---: | ---: | ---: |
| Who is Kane? | 1373 | 1266 | 1781 | 73 |
| What is Edmund's health situation? | 1376 | 1269 | 1784 | 127 |
| Tell me about Edmund | 1373 | 1266 | 1781 | 328 |
| What does Edmund NOT like? | 1375 | 1268 | 1783 | 330 |
| What are Edmund's hardware specs? | 1376 | 1269 | 1784 | 174 |
| When was Edmund born? | 1374 | 1267 | 1782 | 62 |
| What is Edmund's relationship with his mother? | 1378 | 1271 | 1786 | 176 |
| What tools does Edmund use daily? | 1376 | 1269 | 1784 | 189 |
| Summarize Edmund's family | 1375 | 1268 | 1783 | 284 |
| What should I avoid when talking to Edmund? | 1378 | 1271 | 1786 | 176 |

## Test 4: Cross-File Compilation

| File | Source Bytes | Source Tokens | Semantic Bytes | Semantic Tokens | Byte Reduction | Token Reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| USER.md | 4925 | 1338 | 4293 | 1211 | 12.83% | 9.49% |
| SOUL.md | 4485 | 1095 | 4224 | 1065 | 5.82% | 2.74% |
| MEMORY.md | 10402 | 2749 | 9370 | 2508 | 9.92% | 8.77% |
| AGENTS.md | 8088 | 2021 | 7237 | 1866 | 10.52% | 7.67% |
| 2026-03-25.md | 260 | 80 | 239 | 82 | 8.08% | -2.50% |
| 2026-03-26.md | 2145 | 574 | 1911 | 534 | 10.91% | 6.97% |
| 2026-03-27-macmon-heartbeat-milestone.md | 2461 | 693 | 2119 | 614 | 13.90% | 11.40% |
| 2026-03-27-memories-archive.md | 3199 | 861 | 2805 | 762 | 12.32% | 11.50% |
| 2026-03-27-status.md | 3147 | 867 | 2624 | 750 | 16.62% | 13.49% |
| 2026-03-27.md | 3456 | 919 | 2881 | 778 | 16.64% | 15.34% |
| 2026-03-28-famly-discovery.md | 4148 | 1039 | 3679 | 927 | 11.31% | 10.78% |
| 2026-03-28-redis-milestone.md | 1327 | 360 | 1126 | 316 | 15.15% | 12.22% |
| 2026-03-28.md | 5489 | 1515 | 4741 | 1374 | 13.63% | 9.31% |

## Test 5: Fidelity Audit (Expanded)

| File | Record Recall | Negation Recall | Qualifier Recall | Date Recall | Relationship Recall | Quote Fidelity | Path Fidelity | Annotation Fidelity | Probe Recall | Group Focus Purity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| USER.md | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| SOUL.md | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | n/a | n/a | n/a | 100.00% |
| MEMORY.md | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | n/a | 100.00% | 100.00% |
| AGENTS.md | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 92.31% | n/a | n/a | 100.00% | 100.00% |
| 2026-03-25.md | 100.00% | 100.00% | 100.00% | 100.00% | n/a | n/a | n/a | n/a | 100.00% | 100.00% |
| 2026-03-26.md | 100.00% | 100.00% | 100.00% | 100.00% | n/a | n/a | n/a | n/a | 100.00% | 100.00% |
| 2026-03-27-macmon-heartbeat-milestone.md | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | n/a | 100.00% | 100.00% |
| 2026-03-27-memories-archive.md | 91.49% | 100.00% | 100.00% | 73.33% | 100.00% | 92.31% | n/a | n/a | 100.00% | n/a |
| 2026-03-27-status.md | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | n/a | n/a | 100.00% | 100.00% |
| 2026-03-27.md | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | n/a | n/a | 100.00% | 100.00% |
| 2026-03-28-famly-discovery.md | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | n/a | 100.00% | n/a |
| 2026-03-28-redis-milestone.md | 100.00% | 100.00% | 100.00% | 100.00% | n/a | n/a | 100.00% | n/a | 100.00% | n/a |
| 2026-03-28.md | 100.00% | n/a | 100.00% | 100.00% | 100.00% | n/a | 100.00% | n/a | 100.00% | n/a |
