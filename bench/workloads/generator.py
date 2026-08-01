"""
Workload generation for serving benchmarks.

A benchmark's workload is a claim about reality. This module exists to make that
claim explicit, seeded, and — the part that actually matters — MEASURABLE after
the fact, so that a workload which silently degenerated cannot masquerade as the
workload that was requested (docs/RISK_REGISTER.md R14).

WHAT IS GENERATED AND WHY IT IS TOKENS, NOT STRINGS
---------------------------------------------------
Every request is a sequence of TOKEN IDS. Prefix sharing in a radix cache is an
exact match over token ids at block granularity; it is not a match over text. If
this module generated strings, every property it claims — "these two requests
share exactly 8 blocks", "this request diverges 7 tokens into block 5" — would be
a claim about a tokenizer's behaviour rather than about the workload, and would
change when the tokenizer changed. Generating ids makes sharing exact and makes
block-aligned reasoning possible.

`render_to_text()` renders ids back to text through a tokenizer for the HTTP
path, but a tokenizer is NEVER required to construct or analyse a workload, and
the test suite runs without one.

The id space is split so that uniqueness is a construction, not a probability:

    ids [0, marker_space)          MARKER region — a base-`marker_space` counter
    ids [marker_space, vocab_size) FILLER region — random content

Every unique segment begins with a marker encoding a globally unique counter, so
two segments that are supposed to differ differ at a known position, in their
first block, always. Nothing here relies on "random sequences probably don't
collide".

OUTPUT LENGTH IS CONTROLLED, NOT MODEL-DETERMINED
-------------------------------------------------
`Request.max_tokens` is drawn from the output-length distribution and set per
request. docs/BENCHMARK_METHODOLOGY.md §4 commits to this explicitly, and the
reason is comparability across configurations: if output length is whatever the
model decides to emit, then changing the scheduler, the cache, or the batch
composition changes the amount of work in the benchmark, and a throughput delta
becomes uninterpretable. The caller is responsible for the other half of the
contract — suppress EOS, or account for early stops explicitly — and
`WorkloadConfig.suppress_eos` records which was intended. This module cannot
enforce it; it can only make the intent part of the artifact.

THE FOUR PREFIX-SHARING STRUCTURES (§4, "the load-bearing one")
---------------------------------------------------------------
  zero          Every request unique from token 0. THE CONTROL. The cache must
                show ~0% hit rate here, and whatever it costs in latency for
                that zero benefit is the honest floor — §7: "what does the cache
                cost when it never helps?"
  system        A small set of shared system-prompt / few-shot prefixes with
                unique suffixes. The common production shape.
  conversational Turn n contains the ENTIRE history of turn n-1 as a prefix.
                This is the shape that MOST FAVOURS a radix cache and it is
                labeled as such wherever its numbers appear. A hit rate from
                this structure is not comparable to one from `system`.
  adversarial   A long common region, then divergence, with the divergence point
                SWEPT ACROSS EVERY OFFSET WITHIN A BLOCK. Offset 0 is a clean
                block-boundary divergence; offsets 1..block_size-1 are partial
                block matches, which is where partial-hit accounting and
                copy-on-write bugs live. Coverage of the sweep is asserted in
                the realized report, not hoped for.

SHARING RATE IS SWEPT, NEVER FIXED
----------------------------------
§4: "results are reported as a function of sharing rate, never at a single
favorable point." `sweep_sharing_rate()` generates the family. §10 lists the
regimes where prefix-aware routing should be expected to LOSE — zero sharing,
uniformly shared prefixes, skewed prefix popularity — and this generator can
produce all three: `structure="zero"`, `structure="system"` with
`n_shared_prefixes=1`, and `prefix_popularity="zipf"` respectively. A generator
that could not express the losing cases would be a generator that guarantees a
flattering result.

REALIZED VERSUS REQUESTED (R14)
-------------------------------
`Workload.realized` is what the generator ACTUALLY produced: length distribution
with mean/stdev/p50/p90/p99/min/max, clip fractions, realized block-level and
request-level sharing rates, distinct-prefix counts, divergence-offset coverage.
`Workload.degeneracy_warnings` turns the divergences that matter into text.

This is the whole point of the module. A lognormal with a sigma small enough to
collapse toward constant, a sharing rate that produced no sharing because the
shared prefix was shorter than one block, a seeding bug that made every request
identical — all of those produce a benchmark that measures nothing while looking
completely fine in the config. The requested parameters cannot detect any of
them. The realized ones detect all of them.

USAGE
-----
    python3 -m bench.workloads.generator --structure system --sharing-rate 0.8
    python3 -m bench.workloads.generator --structure adversarial --json
    python3 -m bench.workloads.generator --sweep 0,0.25,0.5,0.75,1.0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = [
    "DEFAULT_BLOCK_SIZE",
    "LengthSpec",
    "LengthDraw",
    "Request",
    "Workload",
    "WorkloadConfig",
    "arrival_times",
    "draw_lengths",
    "generate",
    "realized_distribution",
    "render",
    "render_to_text",
    "sweep_sharing_rate",
]

# 16 tokens/block matches serving/memory defaults and the Phase 1 block-boundary
# sweep in results/p1/RESULTS.md. Configurable because the adversarial structure
# is a claim about a specific block size and must be re-swept if it changes.
DEFAULT_BLOCK_SIZE = 16

DEFAULT_VOCAB_SIZE = 32_000
DEFAULT_MARKER_SPACE = 1024
_MARKER_WIDTH = 3          # marker_space**3 == 2**30 distinct segments at defaults

LENGTH_DISTRIBUTIONS = ("lognormal", "uniform", "fixed", "heavy_tail")
SHARING_STRUCTURES = ("zero", "system", "conversational", "adversarial")
ARRIVAL_PROCESSES = ("poisson", "deterministic")


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------


def _substream(seed: int, label: str) -> random.Random:
    """
    A named, independent RNG stream derived from the run seed.

    Separate streams per purpose on purpose: with one shared stream, changing the
    sharing structure would shift the length draws, and a sharing-rate sweep
    would silently also be a length sweep. The two effects would then be
    inseparable in the results. Derived by hashing so the labels can be added to
    without renumbering anything.
    """
    digest = hashlib.blake2b(f"{seed}:{label}".encode(), digest_size=8).digest()
    return random.Random(int.from_bytes(digest, "big"))


# ---------------------------------------------------------------------------
# Length distributions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LengthSpec:
    """
    A stated, seeded length distribution. §4: "stating the distribution is not
    optional; picking a convenient one and not saying so is how projects get
    caught."

    dist:
      lognormal   DEFAULT. Long-tailed, which is what real traffic looks like.
                  `mu` is solved so the unclipped draw has mean `mean`.
      uniform     Symmetric around `mean`, width set by `uniform_spread`.
      fixed       The control. Every draw equals `mean`. Useful precisely
                  because it removes length variance, which makes it the right
                  configuration for isolating a non-length effect — and the
                  wrong one for any headline number.
      heavy_tail  A small fraction of requests get VERY long outputs.

    WHY heavy_tail EXISTS AND WHY IT IS NOT OPTIONAL FOR PREEMPTION WORK
    --------------------------------------------------------------------
    One long generation holding its blocks for thousands of decode steps while
    short requests queue behind it is the exact scenario that motivates
    preemption, fairness, and any admission policy more interesting than FIFO.
    Under a lognormal with a modest sigma, the longest request in a batch is
    only a few times the median, everything drains at roughly the same rate, and
    preemption never has anything to do. A benchmark run only on such a workload
    will show preemption as pure overhead and will conclude, wrongly, that it
    was not worth building. §4 commits to at least one heavy-tailed workload for
    this reason.

    The mean is PRESERVED across the mixture: body mean is solved so that
    (1-f)*body + f*(mult*mean) == mean. That keeps total work roughly constant
    across distributions, so a heavy_tail-vs-lognormal comparison at the same
    requested mean isolates the SHAPE rather than confounding it with load.

    Clipping to [min_len, max_len] is recorded, not assumed harmless: clip
    fractions are part of the realized report because clipping is the specific
    way a heavy tail quietly stops being heavy.
    """

    dist: str = "lognormal"
    mean: int = 256
    sigma: float = 0.8
    uniform_spread: float = 0.9        # fraction of mean; lo = (1-s)m, hi = (1+s)m
    tail_fraction: float = 0.02        # heavy_tail: share of requests in the tail
    tail_multiplier: float = 20.0      # heavy_tail: tail mean = multiplier * mean
    tail_sigma: float = 0.5
    min_len: int = 1
    max_len: int | None = None         # None == unclipped. Set it for a real run.

    def validate(self) -> None:
        if self.dist not in LENGTH_DISTRIBUTIONS:
            raise ValueError(
                f"unknown length distribution {self.dist!r}; "
                f"expected one of {LENGTH_DISTRIBUTIONS}"
            )
        if self.mean <= 0:
            raise ValueError(f"mean must be positive, got {self.mean}")
        if self.min_len < 1:
            raise ValueError(f"min_len must be >= 1, got {self.min_len}")
        if self.max_len is not None and self.max_len < self.min_len:
            raise ValueError(f"max_len {self.max_len} < min_len {self.min_len}")
        if self.dist in ("lognormal", "heavy_tail") and self.sigma <= 0:
            raise ValueError(f"sigma must be positive for {self.dist}, got {self.sigma}")
        if self.dist == "uniform" and not 0.0 <= self.uniform_spread < 1.0:
            raise ValueError(f"uniform_spread must be in [0, 1), got {self.uniform_spread}")
        if self.dist == "heavy_tail":
            if not 0.0 < self.tail_fraction < 1.0:
                raise ValueError(f"tail_fraction must be in (0, 1), got {self.tail_fraction}")
            if self.tail_multiplier <= 1.0:
                raise ValueError(
                    f"tail_multiplier must exceed 1, got {self.tail_multiplier} — "
                    "a tail that is not longer than the body is not a tail"
                )
            if self.tail_fraction * self.tail_multiplier >= 1.0:
                raise ValueError(
                    "tail_fraction * tail_multiplier must be < 1 so the body mean "
                    f"stays positive; got {self.tail_fraction} * {self.tail_multiplier} "
                    f"= {self.tail_fraction * self.tail_multiplier:.3f}"
                )
            if self.tail_sigma <= 0:
                raise ValueError(f"tail_sigma must be positive, got {self.tail_sigma}")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LengthDraw:
    """Drawn lengths plus the clipping that happened on the way out."""

    values: list[int]
    n_clipped_low: int = 0
    n_clipped_high: int = 0

    def clip_fractions(self) -> dict[str, float]:
        n = max(1, len(self.values))
        return {
            "clipped_low": self.n_clipped_low,
            "clipped_high": self.n_clipped_high,
            "clipped_low_fraction": self.n_clipped_low / n,
            "clipped_high_fraction": self.n_clipped_high / n,
        }


def _lognormal_mu(mean: float, sigma: float) -> float:
    """mu such that a lognormal(mu, sigma) has arithmetic mean `mean`."""
    return math.log(mean) - (sigma**2) / 2.0


def draw_lengths(n: int, spec: LengthSpec, rng: random.Random) -> LengthDraw:
    """
    Draw `n` lengths from `spec`, clipped to [min_len, max_len], counting clips.

    The clip counts are not bookkeeping for its own sake. A heavy tail clipped at
    max_len is a heavy tail whose tail has been removed, and the resulting
    workload measures something other than what it says on the label — with no
    error anywhere. That is precisely R14's failure mode, so the count travels
    with the numbers.
    """
    spec.validate()
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")

    raw: list[float] = []
    if spec.dist == "fixed":
        raw = [float(spec.mean)] * n
    elif spec.dist == "uniform":
        lo = spec.mean * (1.0 - spec.uniform_spread)
        hi = spec.mean * (1.0 + spec.uniform_spread)
        raw = [rng.uniform(lo, hi) for _ in range(n)]
    elif spec.dist == "lognormal":
        mu = _lognormal_mu(spec.mean, spec.sigma)
        raw = [rng.lognormvariate(mu, spec.sigma) for _ in range(n)]
    elif spec.dist == "heavy_tail":
        f, mult = spec.tail_fraction, spec.tail_multiplier
        tail_mean = mult * spec.mean
        body_mean = (spec.mean - f * tail_mean) / (1.0 - f)
        body_mu = _lognormal_mu(body_mean, spec.sigma)
        tail_mu = _lognormal_mu(tail_mean, spec.tail_sigma)
        for _ in range(n):
            if rng.random() < f:
                raw.append(rng.lognormvariate(tail_mu, spec.tail_sigma))
            else:
                raw.append(rng.lognormvariate(body_mu, spec.sigma))
    else:  # pragma: no cover - validate() already rejected it
        raise ValueError(f"unknown distribution {spec.dist!r}")

    out = LengthDraw(values=[])
    for x in raw:
        v = int(round(x))
        if v < spec.min_len:
            v = spec.min_len
            out.n_clipped_low += 1
        elif spec.max_len is not None and v > spec.max_len:
            v = spec.max_len
            out.n_clipped_high += 1
        out.values.append(v)
    return out


def _percentile(samples: Sequence[float], q: float) -> float:
    """
    Linear interpolation between order statistics. The method is stated because
    tools disagree and the disagreement shows up exactly in the tail
    (docs/BENCHMARK_METHODOLOGY.md §5). Same rule as bench/capacity.py.
    """
    if not samples:
        raise ValueError("no samples")
    xs = sorted(samples)
    if len(xs) == 1:
        return float(xs[0])
    pos = (len(xs) - 1) * (q / 100.0)
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return float(xs[int(pos)])
    return float(xs[lo] + (xs[hi] - xs[lo]) * (pos - lo))


def realized_distribution(values: Sequence[int]) -> dict[str, Any]:
    """What was ACTUALLY produced. Reported next to every requested parameter."""
    if not values:
        return {"n": 0, "percentile_method": "linear interpolation between order statistics"}
    fl = [float(v) for v in values]
    mean = statistics.fmean(fl)
    stdev = statistics.pstdev(fl) if len(fl) > 1 else 0.0
    p50 = _percentile(fl, 50)
    p99 = _percentile(fl, 99)
    return {
        "n": len(fl),
        "mean": mean,
        "stdev": stdev,
        "cv": (stdev / mean) if mean else 0.0,
        "min": min(values),
        "p50": p50,
        "p90": _percentile(fl, 90),
        "p99": p99,
        "max": max(values),
        # The tail-heaviness statistic. Reported for every distribution so the
        # heavy-tailed workload can be shown to be heavy-tailed rather than
        # asserted to be.
        "p99_over_p50": (p99 / p50) if p50 else float("nan"),
        "percentile_method": "linear interpolation between order statistics",
    }


# ---------------------------------------------------------------------------
# Arrival process
# ---------------------------------------------------------------------------


def arrival_times(
    n: int,
    rate_rps: float,
    process: str = "poisson",
    seed: int = 0,
) -> list[float]:
    """
    Intended dispatch times, seconds from window start.

    Poisson (exponential inter-arrivals) by default per §4: memoryless, produces
    genuine bursts rather than the artificial smoothness of a fixed interval, and
    the burstiness is what stresses admission control. `deterministic` at the
    same mean rate exists so that burstiness can be isolated from load — the gap
    between the two is itself a result about the scheduler.

    These are INTENDED times. Latency is measured from them, not from actual
    send time; that is the coordinated-omission mitigation (§1) and it belongs to
    the load generator, which owns the actual dispatching.
    """
    if process not in ARRIVAL_PROCESSES:
        raise ValueError(f"unknown arrival process {process!r}; expected {ARRIVAL_PROCESSES}")
    if rate_rps <= 0:
        raise ValueError(f"rate_rps must be positive, got {rate_rps}")
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")

    if process == "deterministic":
        step = 1.0 / rate_rps
        return [i * step for i in range(n)]

    rng = _substream(seed, "arrivals")
    t = 0.0
    out = []
    for _ in range(n):
        out.append(t)
        t += rng.expovariate(rate_rps)
    return out


# ---------------------------------------------------------------------------
# Token space
# ---------------------------------------------------------------------------


class _TokenSpace:
    """
    Emits token-id segments with uniqueness by construction.

    Marker ids and filler ids are drawn from disjoint ranges, and every unique
    segment starts with a little-endian base-`marker_space` encoding of a global
    counter. Consequences that the rest of this module depends on:

      * Two unique segments differ within their first `_MARKER_WIDTH` tokens,
        hence within their first block, for any block_size >= 1.
      * A unique segment can never accidentally extend a shared region, because
        its first token is a marker and shared regions are filler.

    Nothing here is probabilistic. "Random token sequences probably don't
    collide" is exactly the kind of assumption that makes a zero-sharing control
    report a nonzero hit rate once in a hundred runs.
    """

    def __init__(self, rng: random.Random, vocab_size: int, marker_space: int) -> None:
        if marker_space < 2:
            raise ValueError(f"marker_space must be >= 2, got {marker_space}")
        if vocab_size <= marker_space:
            raise ValueError(
                f"vocab_size ({vocab_size}) must exceed marker_space ({marker_space}); "
                "filler tokens live above the marker region"
            )
        self.rng = rng
        self.vocab_size = vocab_size
        self.marker_space = marker_space
        self.counter = 0

    def filler(self, n: int) -> list[int]:
        if n <= 0:
            return []
        return [self.rng.randrange(self.marker_space, self.vocab_size) for _ in range(n)]

    def _marker(self, width: int) -> list[int]:
        capacity = self.marker_space**width
        if self.counter >= capacity:
            raise ValueError(
                f"exhausted unique markers: {self.counter} segments requested but a "
                f"{width}-token marker over a {self.marker_space}-id space only "
                f"encodes {capacity}. Raise marker_space or the segment floor."
            )
        c = self.counter
        self.counter += 1
        digits = []
        for _ in range(width):
            digits.append(c % self.marker_space)
            c //= self.marker_space
        return digits

    def unique_segment(self, n: int) -> list[int]:
        """A segment of exactly `n` tokens, globally distinct from every other."""
        if n <= 0:
            raise ValueError(f"unique segment length must be positive, got {n}")
        width = min(_MARKER_WIDTH, n)
        marker = self._marker(width)
        return marker + self.filler(n - width)


# ---------------------------------------------------------------------------
# Config, request, workload
# ---------------------------------------------------------------------------


@dataclass
class WorkloadConfig:
    """
    The requested workload. Everything in here is a claim; `Workload.realized` is
    the audit of that claim.
    """

    n_requests: int = 256
    structure: str = "system"
    sharing_rate: float = 0.5
    block_size: int = DEFAULT_BLOCK_SIZE
    seed: int = 0

    prompt: LengthSpec = field(default_factory=lambda: LengthSpec(mean=512))
    output: LengthSpec = field(default_factory=lambda: LengthSpec(mean=128))

    # structure="system": how many distinct shared prefixes, how long, how popular.
    n_shared_prefixes: int = 4
    shared_prefix_tokens: int = 128           # 8 blocks at block_size=16
    prefix_popularity: str = "uniform"        # "uniform" | "zipf"
    zipf_s: float = 1.1

    # structure="conversational"
    max_turns: int = 8

    # structure="adversarial"
    adversarial_common_tokens: int = 128      # block-aligned common region

    # Token space
    vocab_size: int = DEFAULT_VOCAB_SIZE
    marker_space: int = DEFAULT_MARKER_SPACE

    # Recorded, not enforced here: the caller must actually suppress EOS (or
    # account for early stops) for output length to be controlled (§4).
    suppress_eos: bool = True

    name: str = "workload"

    def validate(self) -> None:
        if self.n_requests < 0:
            raise ValueError(f"n_requests must be non-negative, got {self.n_requests}")
        if self.structure not in SHARING_STRUCTURES:
            raise ValueError(
                f"unknown sharing structure {self.structure!r}; expected {SHARING_STRUCTURES}"
            )
        if not 0.0 <= self.sharing_rate <= 1.0:
            raise ValueError(f"sharing_rate must be in [0, 1], got {self.sharing_rate}")
        if self.block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {self.block_size}")
        if self.n_shared_prefixes < 1:
            raise ValueError(f"n_shared_prefixes must be >= 1, got {self.n_shared_prefixes}")
        if self.shared_prefix_tokens < 1:
            raise ValueError(
                f"shared_prefix_tokens must be >= 1, got {self.shared_prefix_tokens}"
            )
        if self.prefix_popularity not in ("uniform", "zipf"):
            raise ValueError(
                f"unknown prefix_popularity {self.prefix_popularity!r}; expected uniform|zipf"
            )
        if self.max_turns < 1:
            raise ValueError(f"max_turns must be >= 1, got {self.max_turns}")
        if self.adversarial_common_tokens < 1:
            raise ValueError(
                f"adversarial_common_tokens must be >= 1, "
                f"got {self.adversarial_common_tokens}"
            )
        self.prompt.validate()
        self.output.validate()

    def as_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k not in ("prompt", "output")}
        d["prompt"] = self.prompt.as_dict()
        d["output"] = self.output.as_dict()
        return d


@dataclass
class Request:
    """
    One generated request. `token_ids` is the prompt; `max_tokens` is the
    CONTROLLED output length.
    """

    index: int
    token_ids: tuple[int, ...]
    max_tokens: int

    request_id: str = ""
    requested_prompt_len: int = 0     # drawn before any structural floor was applied
    structure: str = "zero"

    # Sharing provenance — what the generator INTENDED. The realized statistics
    # below are computed from the token ids alone and never consult these, so
    # the two are genuinely independent checks on each other.
    shares_prefix_by_construction: bool = False
    shared_prefix_id: int | None = None

    # conversational
    conversation_id: int | None = None
    turn_index: int = 0
    parent_index: int | None = None
    history_len: int = 0
    assistant_tokens: tuple[int, ...] = ()

    # adversarial
    divergence_offset: int | None = None      # offset WITHIN a block, [0, block_size)
    divergence_position: int | None = None    # absolute token position of divergence

    @property
    def prompt_len(self) -> int:
        return len(self.token_ids)

    def as_dict(self, include_tokens: bool = False) -> dict[str, Any]:
        d = {
            "index": self.index,
            "request_id": self.request_id,
            "prompt_len": self.prompt_len,
            "requested_prompt_len": self.requested_prompt_len,
            "max_tokens": self.max_tokens,
            "structure": self.structure,
            "shares_prefix_by_construction": self.shares_prefix_by_construction,
            "shared_prefix_id": self.shared_prefix_id,
            "conversation_id": self.conversation_id,
            "turn_index": self.turn_index,
            "parent_index": self.parent_index,
            "history_len": self.history_len,
            "divergence_offset": self.divergence_offset,
            "divergence_position": self.divergence_position,
        }
        if include_tokens:
            d["token_ids"] = list(self.token_ids)
        return d


@dataclass
class Workload:
    """A generated workload plus its own audit."""

    config: WorkloadConfig
    requests: list[Request]
    realized: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.requests)

    def __iter__(self) -> Iterator[Request]:
        return iter(self.requests)

    @property
    def degeneracy_warnings(self) -> list[str]:
        return list(self.realized.get("degeneracy_warnings", []))

    def fingerprint(self) -> str:
        """
        Content hash over every token id and max_tokens. Two workloads are the
        same workload iff this matches — the reproducibility check in one value,
        and what goes in an artifact when the tokens themselves are too large to
        commit.
        """
        h = hashlib.blake2b(digest_size=16)
        for r in self.requests:
            h.update(b"|")
            h.update(str(r.max_tokens).encode())
            h.update(b":")
            h.update(",".join(str(t) for t in r.token_ids).encode())
        return h.hexdigest()

    def to_dict(self, include_tokens: bool = False) -> dict[str, Any]:
        return {
            "name": self.config.name,
            "requested": self.config.as_dict(),
            "realized": self.realized,
            "fingerprint": self.fingerprint(),
            "requests": [r.as_dict(include_tokens=include_tokens) for r in self.requests],
        }

    def artifact_fields(self) -> dict[str, Any]:
        """
        The dict to hand to `serving.metrics.artifact.Artifact.realized_workload`.

        Deliberately carries BOTH requested and realized. The Artifact docstring
        makes the same point: "divergence between the two is itself the alarm for
        a degenerate workload (R14)". Storing only the realized side would lose
        the comparison; storing only the requested side is the bug.
        """
        return {
            "requested": self.config.as_dict(),
            "realized": self.realized,
            "fingerprint": self.fingerprint(),
        }


# ---------------------------------------------------------------------------
# Sharing statistics — computed from token ids only
# ---------------------------------------------------------------------------


def _sharing_stats(requests: Sequence[Request], block_size: int) -> dict[str, Any]:
    """
    Realized sharing, measured the way the cache would measure it: block-aligned
    prefix matches against everything that arrived earlier.

    `realized_block_sharing_rate` is the ORACLE hit rate — the fraction of
    block-aligned prompt blocks whose exact prefix path was already present in an
    earlier request, i.e. what an infinite, never-evicting cache would hit. It is
    an upper bound on any real cache's hit rate on this workload, which is what
    makes it the right thing to publish next to a measured hit rate: the gap
    between them is eviction and implementation, and without the bound the
    measured number has nothing to be judged against.

    `realized_request_sharing_rate` is the fraction of requests sharing at least
    one full block with an earlier request. This is the statistic comparable to
    the requested `sharing_rate` parameter, and it is structurally slightly lower
    than requested: the FIRST request to use a shared prefix has nobody to share
    with. With k prefixes and n requests the deficit is about k/n.

    Prefix identity is tracked by chained hashing rather than by slicing, so the
    whole pass is linear in total tokens.
    """
    seen: set[int] = set()
    total_blocks = 0
    shared_blocks = 0
    shared_requests = 0
    depths: list[int] = []
    first_blocks: set[tuple[int, ...]] = set()
    full_prompts: set[tuple[int, ...]] = set()

    for r in requests:
        toks = r.token_ids
        n_blocks = len(toks) // block_size
        total_blocks += n_blocks
        full_prompts.add(toks)
        if n_blocks:
            first_blocks.add(toks[:block_size])

        # Walk leading blocks; stop at the first one never seen before.
        acc = 0
        depth = 0
        still_matching = True
        keys = []
        for b in range(n_blocks):
            acc = hash((acc, toks[b * block_size : (b + 1) * block_size]))
            keys.append(acc)
            if still_matching and acc in seen:
                depth += 1
            else:
                still_matching = False
        seen.update(keys)

        shared_blocks += depth
        depths.append(depth)
        if depth > 0:
            shared_requests += 1

    n = len(requests)
    return {
        "block_size": block_size,
        "n_requests": n,
        "total_prompt_blocks": total_blocks,
        "shared_prompt_blocks": shared_blocks,
        "realized_block_sharing_rate": (shared_blocks / total_blocks) if total_blocks else 0.0,
        "realized_request_sharing_rate": (shared_requests / n) if n else 0.0,
        "requests_sharing_at_least_one_block": shared_requests,
        "mean_shared_prefix_blocks": statistics.fmean(depths) if depths else 0.0,
        "max_shared_prefix_blocks": max(depths) if depths else 0,
        "distinct_first_blocks": len(first_blocks),
        "distinct_prompts": len(full_prompts),
        "definition": (
            "block_sharing_rate = leading block-aligned prefix blocks already present "
            "in an earlier request / total block-aligned prompt blocks. This is the "
            "infinite-cache ORACLE hit rate and an upper bound on any measured one."
        ),
    }


def _divergence_stats(requests: Sequence[Request], block_size: int) -> dict[str, Any]:
    """
    Coverage of the adversarial divergence sweep.

    The point of the adversarial structure is that the divergence point lands at
    EVERY offset within a block, because offset 0 (clean block boundary) and
    offsets 1..block_size-1 (partial block) go down different code paths in any
    paged cache — partial-hit accounting, and copy-on-write of a block that is
    shared by refcount but must now diverge. A sweep that happened to cover only
    offsets 0 and 8 would test almost none of that while looking like a sweep, so
    the coverage is measured and published rather than assumed from the
    construction.
    """
    offsets = [r.divergence_offset for r in requests if r.divergence_offset is not None]
    seen = sorted(set(offsets))
    return {
        "n_diverging_requests": len(offsets),
        "offsets_seen": seen,
        "offsets_missing": [o for o in range(block_size) if o not in set(seen)],
        "offset_coverage": len(seen) / block_size if block_size else 0.0,
        "block_boundary_divergences": sum(1 for o in offsets if o == 0),
        "partial_block_divergences": sum(1 for o in offsets if o != 0),
    }


def _degeneracy_warnings(cfg: WorkloadConfig, realized: dict[str, Any]) -> list[str]:
    """
    R14 in executable form: turn requested-vs-realized divergence into text.

    These are warnings, not exceptions. A degenerate workload is sometimes
    exactly what is wanted — `fixed` lengths are a legitimate control — so the
    generator refuses to decide. What it will not do is stay quiet, because
    silence is how a benchmark that measures nothing gets published.
    """
    warnings: list[str] = []
    n = realized.get("n_requests", 0)

    for label, spec in (("prompt", cfg.prompt), ("output", cfg.output)):
        dist = realized.get(f"{label}_len", {})
        if not dist.get("n"):
            continue
        cv = dist.get("cv", 0.0)
        if spec.dist != "fixed" and cv < 0.05:
            warnings.append(
                f"{label} length distribution has COLLAPSED TOWARD CONSTANT: "
                f"requested {spec.dist} mean {spec.mean}, realized cv={cv:.4f} "
                f"(mean {dist['mean']:.1f}, stdev {dist['stdev']:.1f}). A workload "
                "with no length variance removes the phenomena this benchmark "
                "claims to study (R14)."
            )
        clips = realized.get(f"{label}_clipping", {})
        if clips.get("clipped_high_fraction", 0.0) > 0.05:
            warnings.append(
                f"{label} lengths clipped at max_len={spec.max_len} for "
                f"{clips['clipped_high_fraction'] * 100:.1f}% of requests — the "
                "requested tail is not the realized tail."
            )
        if clips.get("clipped_low_fraction", 0.0) > 0.05:
            warnings.append(
                f"{label} lengths clipped at min_len={spec.min_len} for "
                f"{clips['clipped_low_fraction'] * 100:.1f}% of requests."
            )
        if spec.dist == "heavy_tail" and dist.get("p99_over_p50", 0.0) < 3.0:
            warnings.append(
                f"{label} was requested heavy_tail but realized p99/p50 = "
                f"{dist['p99_over_p50']:.2f}, which is not a heavy tail. Preemption "
                "and fairness will look unnecessary on this workload."
            )

    sharing = realized.get("sharing", {})
    if cfg.structure == "zero":
        rate = sharing.get("realized_block_sharing_rate", 0.0)
        if rate > 0.0:
            warnings.append(
                f"structure=zero is the CONTROL and must show no sharing, but the "
                f"realized block sharing rate is {rate:.4f}. Either the generator or "
                "the sharing statistic is wrong; no cache number from this run is "
                "interpretable."
            )
    else:
        requested = cfg.sharing_rate
        realized_req_rate = sharing.get("realized_request_sharing_rate", 0.0)
        # Tolerance: sampling noise ~1/sqrt(n), plus the structural deficit from
        # the first user of each shared prefix having nobody to share with.
        slack = 0.10
        if n:
            slack += cfg.n_shared_prefixes / n + 2.0 / math.sqrt(n)
        if abs(realized_req_rate - requested) > slack:
            warnings.append(
                f"requested sharing_rate={requested:.3f} but realized request-level "
                f"sharing rate is {realized_req_rate:.3f} (tolerance {slack:.3f}). "
                "The workload is not the workload that was asked for."
            )
        if requested > 0.0 and sharing.get("total_prompt_blocks", 0) == 0:
            warnings.append(
                "no prompt is a full block long, so block-granularity sharing cannot "
                f"exist at block_size={cfg.block_size} whatever the sharing rate says."
            )

    if cfg.structure == "system":
        distinct = sharing.get("distinct_first_blocks", 0)
        used = realized.get("distinct_shared_prefixes_used", 0)
        if cfg.sharing_rate >= 1.0 and used < cfg.n_shared_prefixes:
            warnings.append(
                f"requested {cfg.n_shared_prefixes} shared prefixes but only {used} "
                f"appear in the workload ({distinct} distinct first blocks). The "
                "prefix popularity distribution has starved some prefixes."
            )
        if cfg.shared_prefix_tokens < cfg.block_size:
            warnings.append(
                f"shared_prefix_tokens={cfg.shared_prefix_tokens} is shorter than "
                f"block_size={cfg.block_size}: the shared region cannot fill one "
                "block, so a block-granularity cache will see ZERO hits no matter "
                "how high the sharing rate is."
            )

    if cfg.structure == "adversarial":
        div = realized.get("divergence", {})
        if div.get("n_diverging_requests", 0) >= cfg.block_size and div.get(
            "offset_coverage", 0.0
        ) < 1.0:
            warnings.append(
                f"adversarial divergence sweep covers only "
                f"{div['offset_coverage'] * 100:.0f}% of offsets in a "
                f"{cfg.block_size}-token block (missing {div['offsets_missing']}). The "
                "partial-hit and copy-on-write paths this structure exists to "
                "exercise are not all being exercised."
            )

    if n > 1 and sharing.get("distinct_prompts", 0) == 1:
        warnings.append(
            "every request has an IDENTICAL prompt. Almost certainly a seeding bug; "
            "any cache hit rate from this run is meaningless."
        )
    return warnings


# ---------------------------------------------------------------------------
# Structure generators
# ---------------------------------------------------------------------------


def _popularity_weights(k: int, kind: str, s: float) -> list[float]:
    """
    Prefix popularity. `zipf` exists because §10 case 7 — "highly skewed prefix
    popularity" — is named as the most likely place for a genuinely bad routing
    result, and therefore the most valuable one to measure. A generator that
    could only produce uniform popularity could not produce that result at all.
    """
    if kind == "uniform":
        return [1.0] * k
    return [1.0 / ((i + 1) ** s) for i in range(k)]


def _gen_zero(cfg: WorkloadConfig, lens: list[int], space: _TokenSpace) -> list[Request]:
    """
    THE CONTROL. Every request unique from token 0.

    This is what measures what the cache COSTS when it never helps: radix
    insertion, lookup, and refcount bookkeeping on every request for zero
    benefit (§7). `sharing_rate` is ignored here by definition, and that is
    recorded in the realized report rather than silently swallowed.
    """
    return [
        Request(
            index=i,
            token_ids=tuple(space.unique_segment(lens[i])),
            max_tokens=0,
            requested_prompt_len=lens[i],
            structure="zero",
        )
        for i in range(cfg.n_requests)
    ]


def _gen_system(
    cfg: WorkloadConfig, lens: list[int], space: _TokenSpace, rng: random.Random
) -> list[Request]:
    """
    A small set of shared system-prompt / few-shot prefixes with unique suffixes.
    The common production shape.

    Note the special case worth measuring rather than avoiding:
    `n_shared_prefixes=1` at `sharing_rate=1.0` is §10 case 2 — every request
    shares the SAME prefix — where every replica caches it after warmup and
    prefix-aware routing degenerates to "pick any replica". It is the naive
    mental model of prefix caching and the case where the clever router is
    worth nothing.
    """
    prefixes = [
        tuple(space.unique_segment(cfg.shared_prefix_tokens))
        for _ in range(cfg.n_shared_prefixes)
    ]
    weights = _popularity_weights(cfg.n_shared_prefixes, cfg.prefix_popularity, cfg.zipf_s)
    ids = list(range(cfg.n_shared_prefixes))

    out = []
    for i in range(cfg.n_requests):
        want = lens[i]
        if rng.random() < cfg.sharing_rate:
            pid = rng.choices(ids, weights=weights, k=1)[0]
            prefix = prefixes[pid]
            suffix_len = max(_MARKER_WIDTH, want - len(prefix))
            toks = list(prefix) + space.unique_segment(suffix_len)
            out.append(
                Request(
                    index=i,
                    token_ids=tuple(toks),
                    max_tokens=0,
                    requested_prompt_len=want,
                    structure="system",
                    shares_prefix_by_construction=True,
                    shared_prefix_id=pid,
                )
            )
        else:
            out.append(
                Request(
                    index=i,
                    token_ids=tuple(space.unique_segment(want)),
                    max_tokens=0,
                    requested_prompt_len=want,
                    structure="system",
                )
            )
    return out


def _gen_conversational(
    cfg: WorkloadConfig,
    lens: list[int],
    out_lens: list[int],
    space: _TokenSpace,
    rng: random.Random,
) -> list[Request]:
    """
    Multi-turn. Turn n's prompt is turn n-1's prompt PLUS turn n-1's assistant
    tokens PLUS the new user tokens — i.e. it contains the entire history of
    turn n-1 as an exact prefix.

    THIS IS THE MOST FAVOURABLE STRUCTURE FOR A RADIX CACHE and every number
    taken from it is labeled as such. At high sharing rates almost the whole
    prompt of every non-first turn is a cache hit, so hit rates near 1 are the
    expected result and mean nothing on their own; §7: "a 90% hit rate on a
    deep-conversational workload and a 5% hit rate on a zero-sharing workload
    are both correct and neither means anything alone."

    `sharing_rate` is the per-turn continuation probability, so conversation
    length is geometric with mean 1/(1-rate), capped at `max_turns`. At rate 0
    every request is a fresh single-turn conversation and this structure
    degenerates exactly to `zero`, which is the property that makes the sweep
    continuous at its lower end.

    The drawn prompt length is the NEW tokens for the turn; realized prompt
    length grows with turn index. Both are reported, because a realized mean
    prompt length several times the requested one is correct here and would be
    alarming anywhere else.
    """
    out: list[Request] = []
    active: list[int] = []          # indices of requests that may be continued
    next_conv = 0

    for i in range(cfg.n_requests):
        new_tokens = lens[i]
        continue_existing = active and rng.random() < cfg.sharing_rate

        if continue_existing:
            parent_idx = rng.choice(active)
            parent = out[parent_idx]
            history = tuple(parent.token_ids) + tuple(parent.assistant_tokens)
            toks = history + tuple(space.unique_segment(max(_MARKER_WIDTH, new_tokens)))
            req = Request(
                index=i,
                token_ids=toks,
                max_tokens=out_lens[i],
                requested_prompt_len=new_tokens,
                structure="conversational",
                shares_prefix_by_construction=True,
                conversation_id=parent.conversation_id,
                turn_index=parent.turn_index + 1,
                parent_index=parent_idx,
                history_len=len(history),
            )
            active.remove(parent_idx)
        else:
            req = Request(
                index=i,
                token_ids=tuple(space.unique_segment(new_tokens)),
                max_tokens=out_lens[i],
                requested_prompt_len=new_tokens,
                structure="conversational",
                conversation_id=next_conv,
                turn_index=0,
                history_len=0,
            )
            next_conv += 1

        # The assistant reply is synthesised at CONTROLLED length: it is the same
        # max_tokens the request will be served with, so the next turn's history
        # is the history the server would actually have produced.
        req.assistant_tokens = tuple(space.unique_segment(max(1, req.max_tokens)))
        out.append(req)
        if req.turn_index + 1 < cfg.max_turns:
            active.append(i)
    return out


def _gen_adversarial(
    cfg: WorkloadConfig, lens: list[int], space: _TokenSpace, rng: random.Random
) -> list[Request]:
    """
    A long common region, then divergence, with the divergence point swept across
    EVERY offset within a block.

        prompt = common_region  +  straddle_block[:offset]  +  unique_segment

    `common_region` is block-aligned filler shared by the group. `straddle_block`
    is one further block of shared filler, of which the request keeps only the
    first `offset` tokens before the unique segment begins. Because unique
    segments always start in the marker id range and shared regions are always
    filler, divergence happens at EXACTLY position len(common) + offset — the
    property is guaranteed by the id-space split, not by chance.

    Offsets are dealt from a shuffled deck that is refilled every block_size
    draws, so full coverage is guaranteed once block_size requests have
    participated while the order stays unpredictable. Coverage is then MEASURED
    in the realized report anyway; a sweep that silently missed offsets would
    still look like a sweep in the config.

    Two requests in the same group with offsets a < b share
    len(common) + a tokens and diverge at the next one, so the workload contains
    every partial-block match depth from 0 to block_size-1. That is the whole
    point: offset 0 is a clean boundary hit, and everything else forces the cache
    to account for a partially matched block and, if it is doing copy-on-write on
    a block held by more than one sequence, to do it correctly.
    """
    bs = cfg.block_size
    n_groups = cfg.n_shared_prefixes
    commons = [
        tuple(space.filler(cfg.adversarial_common_tokens)) for _ in range(n_groups)
    ]
    straddles = [tuple(space.filler(bs)) for _ in range(n_groups)]
    weights = _popularity_weights(n_groups, cfg.prefix_popularity, cfg.zipf_s)
    gids = list(range(n_groups))

    deck: list[int] = []
    out = []
    for i in range(cfg.n_requests):
        want = lens[i]
        if rng.random() < cfg.sharing_rate:
            g = rng.choices(gids, weights=weights, k=1)[0]
            if not deck:
                deck = list(range(bs))
                rng.shuffle(deck)
            offset = deck.pop()
            head = commons[g] + straddles[g][:offset]
            suffix_len = max(_MARKER_WIDTH, want - len(head))
            out.append(
                Request(
                    index=i,
                    token_ids=head + tuple(space.unique_segment(suffix_len)),
                    max_tokens=0,
                    requested_prompt_len=want,
                    structure="adversarial",
                    shares_prefix_by_construction=True,
                    shared_prefix_id=g,
                    divergence_offset=offset,
                    divergence_position=len(head),
                )
            )
        else:
            out.append(
                Request(
                    index=i,
                    token_ids=tuple(space.unique_segment(want)),
                    max_tokens=0,
                    requested_prompt_len=want,
                    structure="adversarial",
                )
            )
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def generate(cfg: WorkloadConfig) -> Workload:
    """
    Generate a workload and audit it.

    Same seed -> byte-identical workload; different seed -> different workload.
    Both directions are asserted in tests/test_workloads.py, because a generator
    that ignores its seed produces a benchmark whose "independent repetitions"
    are one repetition reported five times.
    """
    cfg.validate()

    len_rng = _substream(cfg.seed, "prompt_lengths")
    out_rng = _substream(cfg.seed, "output_lengths")
    struct_rng = _substream(cfg.seed, f"structure:{cfg.structure}")
    token_rng = _substream(cfg.seed, "tokens")
    space = _TokenSpace(token_rng, cfg.vocab_size, cfg.marker_space)

    prompt_draw = draw_lengths(cfg.n_requests, cfg.prompt, len_rng)
    output_draw = draw_lengths(cfg.n_requests, cfg.output, out_rng)

    if cfg.structure == "zero":
        requests = _gen_zero(cfg, prompt_draw.values, space)
    elif cfg.structure == "system":
        requests = _gen_system(cfg, prompt_draw.values, space, struct_rng)
    elif cfg.structure == "conversational":
        requests = _gen_conversational(
            cfg, prompt_draw.values, output_draw.values, space, struct_rng
        )
    else:
        requests = _gen_adversarial(cfg, prompt_draw.values, space, struct_rng)

    for i, r in enumerate(requests):
        r.max_tokens = output_draw.values[i]     # CONTROLLED, per §4
        r.request_id = f"{cfg.name}-{cfg.seed}-{i:06d}"

    realized = _build_realized(cfg, requests, prompt_draw, output_draw)
    return Workload(config=cfg, requests=requests, realized=realized)


def _build_realized(
    cfg: WorkloadConfig,
    requests: Sequence[Request],
    prompt_draw: LengthDraw,
    output_draw: LengthDraw,
) -> dict[str, Any]:
    """
    The R14 report: what the generator actually produced, computed from the
    generated token ids rather than from the parameters it was given.
    """
    prompt_lens = [r.prompt_len for r in requests]
    out_lens = [r.max_tokens for r in requests]
    sharing = _sharing_stats(requests, cfg.block_size)

    used_prefixes = {r.shared_prefix_id for r in requests if r.shared_prefix_id is not None}
    floored = sum(1 for r in requests if r.prompt_len > r.requested_prompt_len)

    realized: dict[str, Any] = {
        "n_requests": len(requests),
        "structure": cfg.structure,
        "requested_sharing_rate": cfg.sharing_rate,
        "prompt_len": realized_distribution(prompt_lens),
        "prompt_len_requested_mean": cfg.prompt.mean,
        "prompt_len_distribution_requested": cfg.prompt.dist,
        "prompt_clipping": prompt_draw.clip_fractions(),
        "output_len": realized_distribution(out_lens),
        "output_len_requested_mean": cfg.output.mean,
        "output_len_distribution_requested": cfg.output.dist,
        "output_clipping": output_draw.clip_fractions(),
        "output_length_control": (
            "max_tokens set per request from the output distribution; "
            f"suppress_eos={cfg.suppress_eos}. Model-determined output length would "
            "make the workload non-reproducible across configurations (§4)."
        ),
        "sharing": sharing,
        "distinct_shared_prefixes_used": len(used_prefixes),
        "requested_shared_prefixes": (
            cfg.n_shared_prefixes if cfg.structure in ("system", "adversarial") else None
        ),
        "prompt_len_floored_by_structure": floored,
        "prompt_len_floored_fraction": (floored / len(requests)) if requests else 0.0,
        "prompt_len_floored_note": (
            "Requests whose realized prompt is LONGER than the drawn length because a "
            "shared prefix had to fit inside it. Expected and near-total for "
            "`conversational`, where the drawn length is the new tokens of the turn."
        ),
    }

    if cfg.structure == "zero":
        realized["sharing_rate_note"] = (
            "structure=zero ignores sharing_rate by construction: it is the control, "
            "and its realized sharing rate must be exactly 0."
        )
    if cfg.structure == "conversational":
        turns = [r.turn_index for r in requests]
        realized["conversation"] = {
            "n_conversations": len({r.conversation_id for r in requests}),
            "max_turn_index": max(turns) if turns else 0,
            "mean_turn_index": statistics.fmean(turns) if turns else 0.0,
            "continuation_turns": sum(1 for r in requests if r.turn_index > 0),
            "new_tokens_per_turn": realized_distribution(
                [r.requested_prompt_len for r in requests]
            ),
            "note": (
                "MOST FAVOURABLE structure for a radix cache. Realized prompt length "
                "exceeds the requested per-turn length by design — the drawn length is "
                "the NEW tokens of each turn, and history accumulates."
            ),
        }
    if cfg.structure == "adversarial":
        realized["divergence"] = _divergence_stats(requests, cfg.block_size)

    realized["degeneracy_warnings"] = _degeneracy_warnings(cfg, realized)
    return realized


def sweep_sharing_rate(cfg: WorkloadConfig, rates: Sequence[float]) -> list[Workload]:
    """
    The sweep. §4: sharing rate "is a swept parameter, and results are reported as
    a function of sharing rate, never at a single favorable point."

    Every rate reuses the same seed, so length draws are identical across the
    sweep (separate RNG streams) and the only thing that changes is the sharing
    structure. A sweep that also perturbed lengths would confound the two.
    """
    out = []
    for rate in rates:
        sub = WorkloadConfig(**{**cfg.__dict__})
        sub.sharing_rate = rate
        sub.name = f"{cfg.name}_share{rate:.2f}"
        out.append(generate(sub))
    return out


# ---------------------------------------------------------------------------
# Optional text rendering
# ---------------------------------------------------------------------------


def render_to_text(
    workload: Workload, tokenizer: Any = None, model: str | None = None
) -> list[str]:
    """
    Render prompts to text for the HTTP path.

    OPTIONAL BY DESIGN. Generation, analysis, and the entire test suite work on
    token ids and never need a tokenizer; a tokenizer is only required at the
    point where a request has to leave over the wire as JSON. Pass an object with
    `.decode(list[int]) -> str`, or a model name to load one via transformers.

    Caveat that belongs next to the call site: decoding synthetic ids and
    re-encoding them server-side is NOT guaranteed to round-trip to the same ids,
    so a text-rendered workload's realized sharing structure is a property of the
    tokenizer, not of this generator. Prefer submitting token ids directly where
    the API allows it, and if text is used, re-measure sharing after re-encoding.
    """
    if tokenizer is None:
        if model is None:
            raise ValueError(
                "render_to_text needs a tokenizer object or a model name. Workload "
                "generation itself never requires one — token ids are the primary "
                "representation."
            )
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "transformers is not installed, so text rendering is unavailable. "
                "This does not block benchmarking on token ids."
            ) from exc
        tokenizer = AutoTokenizer.from_pretrained(model)
    return [tokenizer.decode(list(r.token_ids)) for r in workload.requests]


# ---------------------------------------------------------------------------
# Reporting — same shape as bench/capacity.py: REQUESTED next to REALIZED
# ---------------------------------------------------------------------------


def _fmt_dist(label: str, requested: str, mean: int, d: dict[str, Any]) -> list[str]:
    if not d.get("n"):
        return [f"  {label:<16} (no samples)"]
    return [
        f"  {label:<16} requested {requested} mean {mean}",
        f"  {'':<16} realized  mean {d['mean']:.1f}  stdev {d['stdev']:.1f}  "
        f"cv {d['cv']:.3f}",
        f"  {'':<16} realized  p50/p90/p99 {d['p50']:.0f} / {d['p90']:.0f} / "
        f"{d['p99']:.0f}   min/max {d['min']} / {d['max']}",
        f"  {'':<16} tail      p99/p50 = {d['p99_over_p50']:.2f}",
    ]


def render(workload: Workload) -> str:
    """Human-readable REQUESTED-vs-REALIZED report, in the house style."""
    cfg, r = workload.config, workload.realized
    sharing = r.get("sharing", {})
    lines = [
        "",
        f"Workload: {cfg.name}   structure={cfg.structure}   seed={cfg.seed}",
        "=" * 78,
        f"  requests          {r['n_requests']}",
        f"  block size        {cfg.block_size}",
        f"  fingerprint       {workload.fingerprint()}",
        "",
        "Length distributions (REQUESTED vs REALIZED)",
    ]
    lines += _fmt_dist(
        "prompt", cfg.prompt.dist, cfg.prompt.mean, r.get("prompt_len", {})
    )
    lines += _fmt_dist(
        "output (max_tok)", cfg.output.dist, cfg.output.mean, r.get("output_len", {})
    )
    pc, oc = r.get("prompt_clipping", {}), r.get("output_clipping", {})
    lines += [
        f"  clipping          prompt low/high {pc.get('clipped_low', 0)}/"
        f"{pc.get('clipped_high', 0)}   output low/high {oc.get('clipped_low', 0)}/"
        f"{oc.get('clipped_high', 0)}",
        f"  length floored    {r.get('prompt_len_floored_by_structure', 0)} "
        f"({r.get('prompt_len_floored_fraction', 0.0) * 100:.1f}%) raised to fit a "
        "shared prefix",
        "",
        "Prefix sharing (REQUESTED vs REALIZED)",
        f"  requested rate    {cfg.sharing_rate:.3f}",
        f"  realized  request-level {sharing.get('realized_request_sharing_rate', 0.0):.3f}"
        f"   block-level (oracle) {sharing.get('realized_block_sharing_rate', 0.0):.3f}",
        f"  blocks            {sharing.get('shared_prompt_blocks', 0)} shared of "
        f"{sharing.get('total_prompt_blocks', 0)}   mean depth "
        f"{sharing.get('mean_shared_prefix_blocks', 0.0):.2f}   max depth "
        f"{sharing.get('max_shared_prefix_blocks', 0)}",
        f"  distinct          first blocks {sharing.get('distinct_first_blocks', 0)}   "
        f"prompts {sharing.get('distinct_prompts', 0)}   shared prefixes used "
        f"{r.get('distinct_shared_prefixes_used', 0)}"
        f" of {r.get('requested_shared_prefixes')}",
    ]

    if "conversation" in r:
        c = r["conversation"]
        lines += [
            "",
            f"  conversations     {c['n_conversations']}   continuation turns "
            f"{c['continuation_turns']}   max turn index {c['max_turn_index']}",
            "  NOTE: most favourable structure for a radix cache; label its numbers.",
        ]
    if "divergence" in r:
        d = r["divergence"]
        lines += [
            "",
            f"  divergence sweep  {d['n_diverging_requests']} diverging requests, "
            f"coverage {d['offset_coverage'] * 100:.0f}% of {cfg.block_size} offsets",
            f"  offsets seen      {d['offsets_seen']}",
            f"  missing           {d['offsets_missing'] or 'none'}",
            f"  boundary/partial  {d['block_boundary_divergences']} / "
            f"{d['partial_block_divergences']}",
        ]

    warnings = r.get("degeneracy_warnings", [])
    lines += ["", "Degeneracy check (R14)"]
    if warnings:
        lines += [f"  !! {w}" for w in warnings]
    else:
        lines += ["  realized matches requested; no degeneracy detected."]
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate a benchmark workload and report REQUESTED vs REALIZED.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-n", "--n-requests", type=int, default=256)
    p.add_argument("--structure", choices=SHARING_STRUCTURES, default="system")
    p.add_argument("--sharing-rate", type=float, default=0.5)
    p.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    p.add_argument("--seed", type=int, default=0, help="RECORDED IN THE ARTIFACT.")
    p.add_argument("--prompt-dist", choices=LENGTH_DISTRIBUTIONS, default="lognormal")
    p.add_argument("--prompt-mean", type=int, default=512)
    p.add_argument("--output-dist", choices=LENGTH_DISTRIBUTIONS, default="lognormal")
    p.add_argument("--output-mean", type=int, default=128)
    p.add_argument("--sigma", type=float, default=0.8)
    p.add_argument("--n-shared-prefixes", type=int, default=4)
    p.add_argument("--shared-prefix-tokens", type=int, default=128)
    p.add_argument("--prefix-popularity", choices=["uniform", "zipf"], default="uniform")
    p.add_argument("--max-turns", type=int, default=8)
    p.add_argument("--sweep", type=str, default=None,
                   help="Comma-separated sharing rates; reports the sweep instead.")
    p.add_argument("--json", action="store_true", help="Emit the realized report as JSON.")
    p.add_argument("--name", type=str, default="workload")
    return p


def _cfg_from_args(args: argparse.Namespace) -> WorkloadConfig:
    return WorkloadConfig(
        n_requests=args.n_requests,
        structure=args.structure,
        sharing_rate=args.sharing_rate,
        block_size=args.block_size,
        seed=args.seed,
        prompt=LengthSpec(dist=args.prompt_dist, mean=args.prompt_mean, sigma=args.sigma),
        output=LengthSpec(dist=args.output_dist, mean=args.output_mean, sigma=args.sigma),
        n_shared_prefixes=args.n_shared_prefixes,
        shared_prefix_tokens=args.shared_prefix_tokens,
        prefix_popularity=args.prefix_popularity,
        max_turns=args.max_turns,
        name=args.name,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = _cfg_from_args(args)

    if args.sweep:
        rates = [float(x) for x in args.sweep.split(",") if x.strip()]
        loads = sweep_sharing_rate(cfg, rates)
        if args.json:
            print(json.dumps([w.artifact_fields() for w in loads], indent=2, sort_keys=True))
            return 0
        print("")
        print(f"Sharing-rate sweep — {cfg.structure}, seed {cfg.seed}, "
              f"n={cfg.n_requests}, block_size={cfg.block_size}")
        print(f"{'requested':>10}  {'realized req-level':>19}  {'realized block-level':>21}"
              f"  {'max depth':>10}")
        for w in loads:
            s = w.realized["sharing"]
            print(f"{w.config.sharing_rate:>10.2f}  "
                  f"{s['realized_request_sharing_rate']:>19.3f}  "
                  f"{s['realized_block_sharing_rate']:>21.3f}  "
                  f"{s['max_shared_prefix_blocks']:>10d}")
        print("\nResults are reported AS A FUNCTION of sharing rate, never at a single "
              "favourable point (§4).\n")
        return 0

    w = generate(cfg)
    if args.json:
        print(json.dumps(w.artifact_fields(), indent=2, sort_keys=True))
    else:
        print(render(w))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
