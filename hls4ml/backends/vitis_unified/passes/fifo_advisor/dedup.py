"""Reduce an LSEnv's FIFO list to the physically distinct channels.

hls4ml's generated wrapper calls the kernel once per input sample, so
LightningSim's trace lists every channel once per invocation: N distinct
channels x S samples entries, all with distinct ids but repeating names. Only
the first N ids are modelled by ``trace_base.compiled``, so FIFO-Advisor's name
map rejects the repeated names ("Duplicate FIFO name found") and its solvers
fail with "no FIFO with id N" on the first repeat.

Keeping the first instance of each name retains exactly the identifiers the
compiled simulation models. That invariant is checked rather than assumed: a
trace of any other shape is refused loudly instead of silently truncated.

This is caller-side: the installed fifo_advisor package is not modified.
"""

from __future__ import annotations

from collections import Counter


class DedupError(RuntimeError):
    pass


def dedup_env(env, *, verify_compiled=True):
    """Deduplicate ``env.fifos`` in place; returns (env, stats_dict)."""
    fifos = list(env.fifos)
    if not fifos:
        raise DedupError('environment lists no FIFOs at all')

    ids = [f.id for f in fifos]
    if len(set(ids)) != len(ids):
        raise DedupError('FIFO ids are not unique; the trace is not a repeated-invocation list')

    counts = Counter(f.name for f in fifos)
    multiplicities = set(counts.values())
    if len(multiplicities) != 1:
        raise DedupError(
            f'channel names do not repeat uniformly (multiplicities {sorted(multiplicities)}); '
            f'this is not the batching-wrapper pattern, refusing to deduplicate'
        )
    (multiplicity,) = multiplicities
    distinct = len(counts)
    if distinct * multiplicity != len(fifos):
        raise DedupError(f'{len(fifos)} FIFOs is not {distinct} channels x {multiplicity} invocations')

    seen, kept = set(), []
    for fifo in fifos:
        if fifo.name in seen:
            continue
        seen.add(fifo.name)
        kept.append(fifo)

    if verify_compiled:
        # The kept ids must be exactly the ones the compiled simulation models;
        # if that is not true, the solvers would fail later and less legibly.
        compiled = env.trace_base.compiled
        unmodelled = []
        for fifo in kept:
            try:
                compiled.get_fifo_design_space([fifo.id], fifo.width)
            except Exception:
                unmodelled.append(fifo.name)
        if unmodelled:
            raise DedupError(
                f'{len(unmodelled)} of the {len(kept)} retained channels have no design '
                f'space in the compiled simulation (e.g. {unmodelled[:3]}); the '
                f'first-occurrence rule did not select the modelled instances'
            )

    env.fifos = kept
    env.num_fifos = len(kept)
    return env, {'total': len(fifos), 'distinct': distinct, 'invocations': multiplicity}


def restrict_env(env, names):
    """Restrict a (deduplicated) env's search space to the named channels.

    The solvers take their optimization variables from ``env.fifos``, so
    dropping a channel here removes it from the search vector. It does NOT
    remove the channel from the model: FIFO-Advisor evaluates a candidate by
    passing it to ``compiled.dse(base_params, design_points)``, where the
    candidate is an *override* on top of the fully as-synthesised parameter
    set. Channels the solvers no longer name therefore keep the depths the
    design was generated with, stay simulated, and still count toward the
    reported latency and BRAM.

    Used by the pass's ``search_scope='compute'`` mode, where only the hls4ml
    inter-layer FIFOs are searched and the wrapper/interface channels are left
    as built.
    """
    names = set(names)
    kept = [f for f in env.fifos if f.name in names]
    if not kept:
        raise DedupError(
            f'restricting to {len(names)} names left no channels; '
            f'the hls4ml variable names do not match the trace channel names'
        )
    env.fifos = kept
    env.num_fifos = len(kept)
    return env, len(kept)
