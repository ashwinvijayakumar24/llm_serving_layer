# Verification of BENCHMARKS.md observations (2026-10-10)

Independent recomputation of every quantitative claim in the Analysis, the W5 paragraph, the interview answers and FINDINGS_OURS F-004/F-006/F-007, from the raw artifacts, with a loader written separately from `bench/xengine/render.py`. Result: 143 claims, 134 match, 8 mismatched (all corrected in commit following this file), 1 not checkable from xengine artifacts (P2 prompt length; resolved from `results/p2/` — it is 128 tokens, and the doc now says so).

# Independent verification of BENCHMARKS.md Analysis / W5 / Interview answers and FINDINGS_OURS F-004, F-006, F-007

Method: own loader (python3, json/glob only) over `results/xengine/W1..W4/<arm>/*_rep*.json` (and `results/xengine_w4_v1/W4/` for F-006's v1 claims). A run counts only if `validity.valid` is true; `ours*` runs also need `validity.attention_backend == "flashinfer"`. Cell value = mean over valid reps. Ratios recomputed from unrounded cell means. Job IDs checked from `hardware.slurm_job_id` (W1 13931401 x210 runs, W2 13931402 x69, W3 13931403 x105, W4 13933751 x45: all MATCH the jobs cited).

Verdicts: MATCH / MISMATCH / NOT-CHECKABLE.

## W5 observation

| # | Claim | Location | Recomputed | Verdict |
|---|---|---|---|---|
| 1 | int8 slower than fp16 at every W1 concurrency | W5 obs | int8 < fp16 at all 7 points (82.0/116.8 ... 1585.9/2110.4) | MATCH |
| 2 | 82 vs 117 tok/s at c1 | W5 obs | 82.02 vs 116.83 | MATCH |
| 3 | −30% at c1 | W5 obs | −29.8% | MATCH |
| 4 | 1586 vs 2110 at c64 | W5 obs | 1585.9 vs 2110.4 | MATCH |
| 5 | −25% at c64 | W5 obs | −24.9% | MATCH |
| 6 | every ours-int8 artifact's server log: 112 linears quantized | W5 obs | both ours-int8 server logs (W1, W2; all artifacts point to one of them) contain `weight_quant=int8 quantized_linears=112` | MATCH |
| 7 | job 13931401 | W5 obs | W1 slurm_job_id 13931401 (W2 int8 runs are 13931402) | MATCH (W1 part) |

## (a) Raw throughput gap

| # | Claim | Location | Recomputed | Verdict |
|---|---|---|---|---|
| 8 | c1: ours 117, vllm 730, sglang 757 tok/s | (a) gap | 116.83 / 729.90 / 757.29 | MATCH |
| 9 | ≈6.2× (vllm/ours) | (a) gap | 6.247 | MATCH |
| 10 | ≈6.5× (sglang/ours) | (a) gap | 6.482 | MATCH |
| 11 | TPOT p99 8.27 vs 1.30 vs 1.25 ms | (a) gap | 8.269 / 1.300 / 1.246 | MATCH |
| 12 | c32 ≈5.5× (vllm) | (a) gap | 10970.3/1983.7 = 5.530 | MATCH |
| 13 | c32 ≈6.0× (sglang) | (a) gap | 11957.3/1983.7 = 6.028 | MATCH |
| 14 | 1984 vs 10970 and 11957 | (a) gap | 1983.7 / 10970.3 / 11957.3 | MATCH |
| 15 | ladder 730 → 185 → 154 | (a) ladder | 729.9 → 184.7 → 154.3 | MATCH |
| 16 | graphs alone divide by ≈3.9 | (a) ladder | 729.9/184.7 = 3.951 (rounds to 4.0) | MISMATCH (borderline; ≈3.95/≈4.0) |
| 17 | removing compile too ≈4.7 | (a) ladder | 729.9/154.3 = 4.730 | MATCH |
| 18 | vllm-eager/ours ≈1.3 | (a) ladder | 154.3/116.8 = 1.321 | MATCH |
| 19 | sglang 757 → sglang-eager 133 (≈5.7×) | (a) SGLang | 757.29/133.19 = 5.686 | MATCH |
| 20 | sglang-eager/ours ≈1.14 (133 vs 117) | (a) SGLang | 1.140 | MATCH |
| 21 | vllm-noasync 477 (−35%) | (a) overlap | 477.1, −34.6% | MATCH |
| 22 | sglang-nooverlap 486 (−36%) | (a) overlap | 486.0, −35.8% | MATCH |
| 23 | c64: vllm 12806, noasync 10977, nograph 8234, eager 7269, ours 2110 | (a) high conc | 12806.5 / 10977.1 / 8234.3 / 7268.8 / 2110.4 | MATCH |
| 24 | graphs ≈1.6× at c64 | (a) high conc | 12806.5/8234.3 = 1.555 | MATCH |
| 25 | "instead of ≈3.9×" | (a) high conc | 3.951 (see #16) | MISMATCH (borderline, ≈4.0) |
| 26 | residual ≈1.3× at batch 1 | (a) reading | 1.321 | MATCH |

## (b) Prefix cache (W3)

| # | Claim | Location | Recomputed | Verdict |
|---|---|---|---|---|
| 27 | job 13931403 | (b) | all W3 runs 13931403 | MATCH |
| 28 | 2–3 valid reps per cell at rates 2–16 | (b) | all 7 arms: 2 or 3 at every rate 2–16 | MATCH |
| 29 | rate 1 has 0–2 valid reps per arm | (b) | ours 1, ours-noprefix 1, vllm 2, sglang/lpm/noradix/vllm-noprefix 0 | MATCH |
| 30 | hit rate 0.62–0.65 for ours, vllm, sglang at every rate (2–16) | (b) | ours 0.618–0.632, vllm 0.629–0.650, sglang 0.635–0.651 (ours rate 1 = 0.610, but rate 1 excluded) | MATCH |
| 31 | 80% share a 1,024-token prefix of ~1,280-token prompt | (b) | W3.yaml sharing_rate 0.8, shared_prefix_tokens 1024, prompt mean 1280; realized request sharing 0.7997, prompt mean 1288 | MATCH |
| 32 | ceiling 0.8×1024/1280 ≈ 0.64 | (b) | 0.640; realized oracle block sharing rate 0.639 | MATCH |
| 33 | units differ (ours blocks, others tokens) | (b) | ours `definitions.prefix_hit_rate` = block granularity | MATCH (ours side) |
| 34 | rate 8 TTFT p50 vllm 15.1 vs vllm-noprefix 17.1 | (b) | 15.06 vs 17.12 | MATCH |
| 35 | sglang 14.8 vs sglang-noradix 15.7 | (b) | 14.78 vs 15.74 | MATCH |
| 36 | saves 1–2 ms | (b) | 2.06 (vllm), 0.95 (sglang) | MATCH |
| 37 | ours rate 2 TTFT p50 84.7 (on) vs 84.3 (off) | (b) | 84.70 vs 84.27 | MATCH |
| 38 | rate 16 p50 728 vs 1,160 | (b) | 728.3 vs 1159.7 | MATCH |
| 39 | rate 16 p99 1,730 vs 2,390 | (b) | 1726.4 vs 2386.0 | MATCH |
| 40 | sglang-lpm matches sglang "within noise at every rate" | (b) | LPM TTFT p50 is higher at every rate: 15.90/14.75 (r2), 15.93/14.85 (r4), 15.46/14.78 (r8), 16.71/15.38 (r16), i.e. +0.7 to +1.3 ms (+5–9%); per-rep ranges do not overlap at rates 2, 4, 16 (sglang r2 14.73–14.75 vs lpm 14.99–17.44; r4 14.74–14.95 vs 15.10–16.75; r16 15.22–15.53 vs 16.40–17.02) | MISMATCH (qualitative: LPM is slightly but consistently slower; "adds nothing" still holds) |
| 41 | rate 8: lpm 15.5 vs sglang 14.8 | (b) | 15.46 vs 14.78 | MATCH |
| 42 | ours W3 TPOT p50 35–72 ms (rates 2–16) | (b) | 35.3 / 48.4 / 53.8 / 72.1 | MATCH |
| 43 | "vs 8.3–11.4 ms on W1" (TPOT p50) | (b) | W1 ours TPOT p50 = 8.17–21.58 over c1–c64 (8.17–10.93 for c1–c16). 8.27–11.41 is TPOT **p99** at c1–c16 | MISMATCH |
| 44 | ours goodput 0 from rate 8 up | (b) | r8 0.0, r16 0.0 (r4 0.408) | MATCH |
| 45 | TPOT SLO 27.8 ms | (b) | slo.tpot_ms 27.8 | MATCH |
| 46 | vLLM/SGLang TPOT p50 1.2–1.5 ms on W3 | (b) | 1.232–1.507 | MATCH |

## (c) Preemption (W4 v2)

| # | Claim | Location | Recomputed | Verdict |
|---|---|---|---|---|
| 47 | job 13933751 | (c) | all 45 W4 runs 13933751 | MATCH |
| 48 | "all 45 cells valid, 3/3 repetitions each" | (c) | 5 arms × 3 points = **15 cells**, 45 runs, all valid, 3/3 each | MISMATCH (45 runs, 15 cells) |
| 49 | same 32,768-token pool for every engine | (c) | ours/ours-noprefix kv_blocks 2048×16; vllm num_gpu_blocks 2048; sglang resolved max_total_num_tokens 32768 | MATCH |
| 50 | ~2.5k-token sequences | (c) | prompt mean 2072 + 512 output = ~2.58k | MATCH |
| 51 | 64 measured at c16, 128 at c32 | (c) | steady_requests 64 / 128 | MATCH |
| 52 | c16 ours 1,781–1,935 | (c) | 1781, 1832, 1935 | MATCH |
| 53 | c16 ours-noprefix 890–1,092 | (c) | 890, 937, 1092 | MATCH |
| 54 | c16 vllm 21–22 | (c) | 22, 22, 21 | MATCH |
| 55 | c16 sglang 4–5 | (c) | 5, 5, 4 | MATCH |
| 56 | c32 ours 3,820–4,141 | (c) | 3820, 3901, 4141 | MATCH |
| 57 | c32 ours-noprefix 3,467–4,031 | (c) | 3605, 4031, 3467 | MATCH |
| 58 | c32 vllm 43–44 | (c) | 43, 44, 44 | MATCH |
| 59 | c32 sglang 8–10 | (c) | 8, 9, 10 | MATCH |
| 60 | c16 tok/s vllm 5,786, sglang 5,551, ours-noprefix 874, ours 592 | (c) | 5785.9 / 5550.9 / 873.8 / 592.4 | MATCH |
| 61 | c32 vllm 5,448, sglang 5,472 ("hold their throughput") | (c) | 5447.5 / 5471.9 (−5.8% and −1.4% vs c16) | MATCH |
| 62 | c32 TTFT p50 1.1–1.4 s (vllm/sglang) | (c) | 1.100 / 1.397 s | MATCH |
| 63 | 108,156 tokens recomputed in a single c4 run | (c) | ours c4 rep1 tokens_recomputed 0 → 108,156 | MATCH |
| 64 | demand ≈10k of 32k at c4 | (c) | 4 × ~2.58k ≈ 10.3k | MATCH |
| 65 | c4: vllm, sglang, ours-noprefix preempt 0 | (c) | 0 in all 9 runs | MATCH |
| 66 | c4 ours 111–143 | (c) | 111, 143, 141 | MATCH |
| 67 | ours ends c4 point with every block held by cache | (c) | rep1 after: blocks_free 0, cached 2048; rep2 after: 0 / 2048; rep3 after: free 4, cached 2044 | MATCH (rep3 99.8%) |
| 68 | SGLang evicts 114k–532k tokens per run | (c) | 114,166 – 531,985 | MATCH |
| 69 | SGLang retracted ≤10 | (c) | max 10 (c32 rep3) | MATCH |
| 70 | vllm-matched 5,776 vs 5,786 at c16 | (c) | 5776.4 vs 5785.9 | MATCH |
| 71 | goodput 0 for every engine at c32 | (c) | all five arms 0.0 | MATCH |
| 72 | TTFT SLO 434 ms | (c) | slo.ttft_ms 434 | MATCH |
| 73 | c16 goodput vllm 9.8, sglang 7.4, ours 0.24, ours-noprefix 0.76 | (c) | 9.843 / 7.388 / 0.242 / 0.760 | MATCH |
| 74 | ≈30 preemptions per completed request at c32 | (c) reading | ours mean 3,954 preemptions per point. The counter is a delta over the whole point, during which the server completed 192 requests (incl. warmup) → **20.6**. Only if divided by measured-window completions (125.3) is it 31.6 | MISMATCH (≈21 per request processed; ≈32 only with the measured-window denominator, which undercounts the requests that generated the preemptions) |

## (d) Latency tails

| # | Claim | Location | Recomputed | Verdict |
|---|---|---|---|---|
| 75 | job 13931402 | (d) W2 | all W2 runs 13931402 | MATCH |
| 76 | 1–3 valid reps per cell | (d) W2 | valid cells have 1–3 (vllm r32 = 3) | MATCH |
| 77 | none for any arm at rate 2 | (d) W2 | 0 valid for ours, ours-int8, vllm, sglang | MATCH |
| 78 | none for ours at rates 1 and 32 | (d) W2 | ours valid only at 4, 8, 16 | MATCH |
| 79 | vLLM and SGLang 100% SLO at every rate with valid data | (d) W2 | slo_attainment 1.0 in every valid cell (r1,4,8,16,32) | MATCH |
| 80 | up to 32 rps, goodput 32.8 and 33.2 | (d) W2 | vllm 32.80, sglang 33.20 | MATCH |
| 81 | ours ≈1% SLO at rates 4–16 | (d) W2 | 0.010 / 0.014 / 0.011 | MATCH |
| 82 | ours TPOT p50 48–64 ms | (d) W2 | 47.6 / 52.0 / 63.7 | MATCH |
| 83 | TPOT SLO 27.8 | (d) W2 | 27.8 | MATCH |
| 84 | SLO calibrated on P2 at 256-token mean prompts | (d) W2 | slo.source cites `results/p2/RESULTS.md (job 11608159 ...)` but no prompt length is in the artifacts | NOT-CHECKABLE (P2 prompt length not in xengine artifacts) |
| 85 | W2 512-token lognormal prompts | (d) W2 | W2.yaml lognormal mean 512; realized mean 508.8 | MATCH |
| 86 | TPOT p50 8.3 ms at W1 c1 | (d) W1/W3 | p50 = **8.17** (8.2); 8.27 is p99 | MISMATCH |
| 87 | 35 ms at W3 rate 2, 72 ms at rate 16 | (d) W1/W3 | 35.3 / 72.1 | MATCH |
| 88 | vLLM/SGLang ~1.3 → ~1.5 ms over same range | (d) W1/W3 | W1 c1 1.27/1.23 → W3 r16 1.51/1.50 | MATCH |

## (e) Scaling shape

| # | Claim | Location | Recomputed | Verdict |
|---|---|---|---|---|
| 89 | c32 normalized: ours 16.98×, vllm 15.03×, sglang 15.79× | (e) | 16.979 / 15.030 / 15.790 | MATCH |
| 90 | ours ≥ both baselines up to c32 | (e) | ours higher at c2,4,8,16,32 (1.93 vs 1.85/1.85 ... 16.98 vs 15.03/15.79) | MATCH |
| 91 | ours 16.98× → 18.06× | (e) | 18.063 | MATCH |
| 92 | vllm 15.03× → 17.55× | (e) | 17.546 | MATCH |
| 93 | 1984 → 2110 tok/s | (e) | 1983.7 → 2110.4 | MATCH |
| 94 | TTFT p99 602 → 2233 ms | (e) | 602.1 → 2233.1 | MATCH |
| 95 | our batch cap 32 | (e) reading | artifact flags max_batch_size 32 | MATCH |
| 96 | vllm-matched c64 11745 vs 12806 | (e) | 11745.1 vs 12806.5 | MATCH |
| 97 | TTFT p99 350 → 448 ms | (e) | 350.3 → 447.5 | MATCH |
| 98 | cap costs vLLM ≈8% | (e) | 8.29% | MATCH |

## Five interview questions

| # | Claim | Location | Recomputed | Verdict |
|---|---|---|---|---|
| 99 | gap ≈6.2× at batch 1 | Q1 | 6.247 | MATCH |
| 100 | graphs+compile off removes ≈4.7× (730 → 154) | Q1 | 4.730 | MATCH |
| 101 | leaves ≈1.3× to ours (117) | Q1 | 1.321 | MATCH |
| 102 | overlap worth ≈35% in both | Q1 | −34.6% / −35.8% | MATCH |
| 103 | graphs ≈1.6× at c64 | Q1 | 1.555 | MATCH |
| 104 | ceiling ≈0.63 hit rate | Q2 | cells 0.618–0.651, oracle 0.639 | MATCH |
| 105 | latency payoff 1–2 ms vLLM/SGLang | Q2 | 2.06 / 0.95 | MATCH |
| 106 | ours preempted ≈1,850 per run at 16 in flight | Q3 | mean 1,849 | MATCH |
| 107 | vs 22 (vLLM) and 5 (SGLang) | Q3 | 21.7 / 4.7 | MATCH |
| 108 | 592 vs ≈5,600 tok/s | Q3 | ours 592.4; vllm 5,786, sglang 5,551 (mean of the two 5,668 → ≈5,700) | MISMATCH (minor; say "≈5,550–5,790" or "≈5,700") |
| 109 | 32k-token pool | Q3 | 32,768 for all arms | MATCH |

## FINDINGS_OURS F-004

| # | Claim | Recomputed | Verdict |
|---|---|---|---|
| 110 | job 13933751, c4 demand ≈10k of 32,768 | see #47, #64 | MATCH |
| 111 | ours preempted 111, 143, 141 | 111, 143, 141 | MATCH |
| 112 | evicted 5,975–8,016 cached blocks | 5975, 7611, 8016 | MATCH |
| 113 | ours-noprefix preempted 0 in all three | 0, 0, 0 | MATCH |
| 114 | cache-on run ended blocks_free=0, cache_cached_blocks=2048, admission_control_alarm=True | c4 rep1 after: 0 / 2048 / True (False before) | MATCH |
| 115 | 108,156 tokens recomputed | c4 rep1 0 → 108,156 | MATCH |

## FINDINGS_OURS F-006

| # | Claim | Recomputed | Verdict |
|---|---|---|---|
| 116 | v1 evidence job 13931404 | all v1 ours runs 13931404 | MATCH |
| 117 | rate2_rep1 and rate1_rep2 completed 0 of 186 and 86; all no_content | 0/186 and 0/86 measured, outcomes no_content 186 / 86 | MATCH |
| 118 | before rate2_rep1: running=0 waiting=0 blocks_free=0 blocks_used=2048 | exact | MATCH |
| 119 | 772 steps, 16,512 cache blocks evicted | 772 / 16,512 | MATCH |
| 120 | no first token for any of 256 requests | 256 received; server output_tokens_total delta 0 | MATCH |
| 121 | starvation_fallbacks 1,619 before rate2_rep1, 105,769 before rate1_rep2 | 1,619 / 105,769 | MATCH |
| 122 | rate4_rep1 from same full-cache state made progress | before free 0/used 2048; 361/361 completed | MATCH |
| 123 | 1,991 preemptions at 1 req/s | rate1_rep1 1,991 | MATCH |
| 124 | vLLM and SGLang preempted 0 at rate 1 (v1) | 0 in all 3 reps each | MATCH |
| 125 | W2 pool 277,590 blocks | allocator num_blocks 277,590 | MATCH |
| 126 | W2 before snapshots running=0 waiting=0 every time | all 18 cells | MATCH |
| 127 | blocks_free 277,590 → 223,571 → 167,282 → 75,304 → 49,624 → 20,106 → 0 | all seven values present in chronological order (rate1_rep1, rate16_rep1, rate32_rep1, rate1_rep2, rate8_rep2, rate16_rep2, rate32_rep2); it is a selection of 7 of 18 cells, full sequence also monotone falling to 0 then 0–27 in rep 3 | MATCH (selected subsequence) |
| 128 | goodput 0.00 at every rate in rep 3 | all six rep-3 cells 0.0 | MATCH |
| 129 | rate 1 was 0.90 in rep 1 | 0.900 (run invalid, but value as stated) | MATCH |
| 130 | 26 of 2,013 completed in rate32_rep2 | 26 / 2013 | MATCH |
| 131 | 0 of 1,920 in rate32_rep3 | 0 / 1920 | MATCH |
| 132 | vLLM shows no drift over the same 18 cells | vllm:kv_cache_usage_perc 0.0 and running/waiting 0 before every cell; goodput stable (rate32 32.85/33.55/32.0) | MATCH |
| 133 | W3: 15 cells, ours 277,590 → 20,634 free at start of last cell | 15 cells; last (rate16_rep3) before free 20,634 | MATCH |
| 134 | ours-noprefix stayed exactly 277,590 | all 15 | MATCH |
| 135 | W1 ours never below 87,166 (31% free) | min 87,166 (concurrency64_rep3) = 31.4% | MATCH |
| 136 | W4 v2: fresh ours server hit blocks_free=0, all cached, during first point (c4) | c4 rep1 before 2048 free → after 0 free, 2048 cached | MATCH |
| 137 | started c16 and c32 points with pool full of cache (flagged) | pre_point_state free 0/0/4 (c16), 0/34/0 (c32), all <5% of 2048; flag present in W4 tables | MATCH (c4 cell is also flagged because reps 2–3 began full) |
| 138 | ours-noprefix started every point with all 2,048 free | all 9 runs 2048 | MATCH |

## FINDINGS_OURS F-007

| # | Claim | Recomputed | Verdict |
|---|---|---|---|
| 139 | 12, 15, 18 of 48 measured no_content (reps 1–3) | 12 / 15 / 18 of 48 | MATCH |
| 140 | requests_completed 56 of 56 received (rep 1) | 0→56 received, 0→56 completed | MATCH |
| 141 | 20,330 output tokens vs 28,672 expected (56 × 512) | 20,330; 56×512 = 28,672 (W4 output fixed 512) | MATCH |
| 142 | ours-noprefix emitted 28,486, 0 no_content | 28,486; 0 no_content in all reps | MATCH |
| 143 | starvation_fallbacks rose by 13 | 0 → 13 | MATCH |

## Totals

143 claims: 134 MATCH, 8 MISMATCH (rows 16, 25, 40, 43, 48, 74, 86, 108), 1 NOT-CHECKABLE (row 84).
