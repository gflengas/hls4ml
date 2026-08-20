"""Run FIFO-Advisor's solvers over a synthesised HLS solution and pick a result.

Selection is over the Pareto front of *every* solver's evaluations pooled
together, not any single solver's best. That matters: on the MADOS 10k UNet
only the heuristic search finds the skip-connection structure (221 BRAM
estimated), while the annealing and random searches return 3045 and 2562 BRAM
-- assignments that are deadlock-free but correspond to synthesised designs at
127% and 115% of a ZCU102. On the MNIST-ladder models the ordering reverses on
two models. No single strategy dominates; the front does.
"""

from __future__ import annotations

import time

from . import ls_compat
from .dedup import dedup_env, restrict_env

DEFAULT_SOLVERS = ('heuristic', 'sa', 'group-sa', 'random', 'group-random')


class AdvisorResult:
    """Outcome of a sweep: the chosen depths plus everything needed to audit them."""

    def __init__(self, depths, winner, front, per_solver, fifo_stats, compat_stats, timings):
        self.depths = depths
        self.winner = winner
        self.front = front
        self.per_solver = per_solver
        self.fifo_stats = fifo_stats
        self.compat_stats = compat_stats
        self.timings = timings

    def to_dict(self):
        """Serialise the result for ``fifo_depths_advisor.json``."""
        return {
            'winner': self.winner,
            'pareto_front': self.front,
            'solvers': self.per_solver,
            'fifo_stats': self.fifo_stats,
            'ls_compat_stats': self.compat_stats,
            'timings_s': self.timings,
            'depths': self.depths,
        }


def _solver_classes():
    from fifo_advisor.solvers import (
        DiscreteSimulatedAnnealingOptimizer,
        GroupedDiscreteSimulatedAnnealingOptimizer,
        GroupRandomSearchOptimizer,
        HeuristicOptimizer,
        RandomSearchOptimizer,
    )

    return {
        'heuristic': HeuristicOptimizer,
        'sa': DiscreteSimulatedAnnealingOptimizer,
        'group-sa': GroupedDiscreteSimulatedAnnealingOptimizer,
        'random': RandomSearchOptimizer,
        'group-random': GroupRandomSearchOptimizer,
    }


def select_from_front(front, selection):
    """Pick one assignment off the combined Pareto front.

    ``min_bram`` is the default because it is the rule that was verified
    against real RTL: on the MADOS UNet the min-BRAM point co-simulates to
    Pass and fits a ZCU102 at 13%, while the stochastic solvers' own bests do
    not fit at all.
    """
    if callable(selection):
        return selection(front)
    if selection == 'min_bram':
        return min(front, key=lambda r: (r.bram_usage_total, r.latency))
    if selection == 'min_latency':
        return min(front, key=lambda r: (r.latency, r.bram_usage_total))
    raise ValueError(f'unknown selection rule: {selection!r}')


def analyze_solution(
    solution_dir,
    *,
    solvers=DEFAULT_SOLVERS,
    selection='min_bram',
    restrict_names=None,
    retrace=True,
    verbose=True,
):
    """Trace, deduplicate, sweep and select. Returns an :class:`AdvisorResult`.

    If ``restrict_names`` is given, the solvers' search space is limited to
    channels with those names after deduplication; every other channel keeps
    the depth the design was generated with (the solvers never vary it, but it
    remains part of the simulated design and of the reported cost).
    """
    import asyncio
    from pathlib import Path

    from lightningsim.model.solution import Solution
    from lightningsim.runner import Runner

    from fifo_advisor.main import collect_baseline_results, fifo_id_to_name_map_from_env
    from fifo_advisor.opt_env import LSEnv, is_pareto_efficient_simple

    solution_dir = Path(solution_dir)
    timings = {}

    if retrace:
        # LightningSim caches a resolved trace next to the solution; a stale one
        # would describe a previous build of the project.
        (solution_dir / 'trace.pkl').unlink(missing_ok=True)

    t0 = time.time()
    asyncio.run(Runner(Solution(solution_dir), debug=True).run())
    timings['trace'] = round(time.time() - t0, 3)
    if verbose:
        print(f'  trace resolution : {timings["trace"]}s')

    t0 = time.time()
    env, fifo_stats = dedup_env(LSEnv(solution_dir))
    if restrict_names is not None:
        env, n_kept = restrict_env(env, restrict_names)
        fifo_stats['searched'] = n_kept
    timings['env'] = round(time.time() - t0, 3)
    if verbose:
        searched = fifo_stats.get('searched', fifo_stats['distinct'])
        print(
            f'  env construction : {timings["env"]}s '
            f'({fifo_stats["distinct"]} channels x {fifo_stats["invocations"]} invocations '
            f'= {fifo_stats["total"]} entries; searching {searched})'
        )

    names = fifo_id_to_name_map_from_env(env)

    def named(sizes):
        return {names[f]: int(v) for f, v in sizes.items() if f in names}

    everything, per_solver, solver_of = [], {}, {}
    classes = _solver_classes()

    t_sweep = time.time()
    for label in ('baseline', *solvers):
        t0 = time.time()
        try:
            evals = collect_baseline_results(env) if label == 'baseline' else classes[label](env).solve()
        except Exception as exc:  # one solver failing must not abort the sweep
            per_solver[label] = {'failed': f'{type(exc).__name__}: {exc}'}
            if verbose:
                print(f'  {label:<13} FAILED - {type(exc).__name__}: {exc}')
            continue
        dt = round(time.time() - t0, 3)
        ok = [r for r in evals if not r.deadlock]
        if not ok:
            per_solver[label] = {'runtime_s': dt, 'n_evals': len(evals), 'all_deadlock': True}
            if verbose:
                print(f'  {label:<13} runtime={dt:7.3f}s  evals={len(evals):>4}  all deadlock')
        else:
            best = min(ok, key=lambda r: (r.bram_usage_total, r.latency))
            per_solver[label] = {
                'runtime_s': dt,
                'n_evals': len(evals),
                'best_latency': best.latency,
                'best_bram': best.bram_usage_total,
                # The depths this solver would have shipped had it been the only
                # one run. Without these the report answers "how good was each
                # solver" but not "what would it actually have applied", which is
                # the question worth asking when deciding to run just one.
                'best_depths': named(best.fifo_sizes),
            }
            if verbose:
                print(
                    f'  {label:<13} runtime={dt:7.3f}s  evals={len(evals):>4}  '
                    f'best latency={best.latency} bram={best.bram_usage_total}'
                )
        # Tag each evaluation with the solver that produced it. The front is
        # pooled across solvers, so without this the shipped depths cannot be
        # attributed afterwards -- "which strategy produced what I am building"
        # is not answerable from the report. Recorded by identity rather than
        # by wrapping the result, so nothing downstream that consumes an
        # evaluation object has to change.
        for _r in evals:
            solver_of[id(_r)] = label
        everything.extend(evals)
    timings['sweep'] = round(time.time() - t_sweep, 3)

    nd = [r for r in everything if not r.deadlock]
    if not nd:
        raise RuntimeError('every evaluated depth assignment deadlocked; nothing to select')

    flags = is_pareto_efficient_simple(nd)
    seen, front = set(), []
    for r in sorted((r for r, f in zip(nd, flags) if f), key=lambda r: (r.bram_usage_total, r.latency)):
        if (r.latency, r.bram_usage_total) not in seen:
            seen.add((r.latency, r.bram_usage_total))
            front.append(r)

    # Several solvers routinely land on the same (latency, bram) point. Report
    # every solver that reached it, cheapest-first in sweep order, rather than
    # crediting only whichever happened to be evaluated first: on these models
    # the greedy heuristic and the stochastic solvers often agree, and hiding
    # that would misrepresent how much the expensive solvers contributed.
    def solvers_reaching(result):
        key = (result.latency, result.bram_usage_total)
        found = [
            lbl
            for lbl in ('baseline', *solvers)
            if any(solver_of.get(id(r)) == lbl and (r.latency, r.bram_usage_total) == key for r in nd)
        ]
        return found or ['unknown']

    win = select_from_front(front, selection)
    if verbose:
        print(f'  pareto front     : {[(r.latency, r.bram_usage_total) for r in front]}')
        for r in front:
            print(f'    ({r.latency}, {r.bram_usage_total}) found by: {", ".join(solvers_reaching(r))}')
        print(f'  WINNER           : latency={win.latency} bram={win.bram_usage_total}')
        print(f'  winner found by  : {", ".join(solvers_reaching(win))}')

    return AdvisorResult(
        depths=named(win.fifo_sizes),
        winner={
            'latency': win.latency,
            'bram': win.bram_usage_total,
            'found_by': solvers_reaching(win),
        },
        # Each front point carries its depths, not just its cost: a reviewer
        # comparing the shipped assignment against the alternatives needs to
        # see what the alternatives actually were.
        front=[
            {
                'latency': r.latency,
                'bram': r.bram_usage_total,
                'found_by': solvers_reaching(r),
                'depths': named(r.fifo_sizes),
            }
            for r in front
        ],
        per_solver=per_solver,
        fifo_stats=fifo_stats,
        compat_stats=ls_compat.stats(),
        timings=timings,
    )
