"""Static-analysis FIFO-depth optimization (FIFO-Advisor + LightningSim).

This is the analysis-based counterpart to ``vitisunified:fifo_depth_optimization``.
That pass inflates every FIFO to 100,000, runs C synthesis **and RTL
co-simulation** of the inflated design, and reads each channel's peak occupancy
out of the co-simulation's ``channel.zip``. This pass synthesises once at the
*default* depths and then searches statically over LightningSim's model of the
design; no co-simulation is run.

The two answer different questions. Co-simulation measures the occupancy that
naturally accumulates when space is ample; the static search finds the smallest
depth that avoids a stall. Both passes end by calling the same
:func:`set_optimized_fifo_depths`, so they are interchangeable from the
caller's point of view and exactly one of them should be in ``config['Flows']``.

Usage::

    ADVISOR = 'vitisunified:fifo_depth_optimization_advisor'
    config['Flows'] = [ADVISOR]
    hls4ml.model.optimizer.get_optimizer(ADVISOR).configure(
        selection='min_bram',  # 'min_bram' | 'min_latency' | callable(front) -> point
        search_scope='compute',  # 'compute' | 'all'
        solvers=('heuristic', 'sa', 'group-sa', 'random', 'group-random'),
        write_report=True,  # fifo_depths_advisor.json next to fifo_depths.json
        skip_synthesis=False,  # reuse an already-synthesised solution
    )

Every option is shown with its default. See :class:`FifoDepthOptimizationAdvisor`
for what each one means.

What the pass does, in order (all inside :meth:`FifoDepthOptimizationAdvisor.transform`)::

    1. validate IOType / search_scope / selection / solver names   -- before any build
    2. check lightningsim + fifo_advisor are importable            -- before any build
    3. ls_compat.install()          patch LightningSim in memory, before it is imported
    4. model.write(); model.build(synth=True, cosim=False)         C synthesis at DEFAULT depths
    5. hls_app.generate() + verify()                               reconstruct hls.app for LightningSim
    6. collect the hls4ml stream variables that may receive a depth
    7. sweep.analyze_solution()     trace -> dedup -> restrict -> solvers -> Pareto front -> select
    8. apply only if it beats the defaults, then generate_depths_file()
    9. write fifo_depths_advisor.json

Requires ``lightningsim`` and ``fifo_advisor`` in the same interpreter. That
restricts the flow to **linux-64 on CPython 3.11 or 3.12**: LightningSim's conda
channel publishes linux-64 only, with builds per interpreter version, and
``fifo_advisor`` requires Python >= 3.11. Neither is on PyPI, so neither can be
declared as an hls4ml dependency; both are imported lazily inside ``transform``
so that hls4ml still imports for users who do not have them, and their absence
is reported before any synthesis runs.
"""

import importlib.util
import json
import os
import time

from hls4ml.backends.vitis.passes.fifo_depth_optimization import (
    generate_depths_file,
    set_optimized_fifo_depths,
)
from hls4ml.model.optimizer.optimizer import ConfigurableOptimizerPass, ModelOptimizerPass

_KNOWN_SOLVERS = ('heuristic', 'sa', 'group-sa', 'random', 'group-random')


class FifoDepthOptimizationAdvisor(ConfigurableOptimizerPass, ModelOptimizerPass):
    """Static-analysis alternative to ``vitisunified:fifo_depth_optimization``.

    Configure through ``get_optimizer(<flow>).configure(...)``:

    solvers (tuple[str]):
        Which FIFO-Advisor strategies to run. Any subset of
        ``('heuristic', 'sa', 'group-sa', 'random', 'group-random')``; the
        depth-2 and default-depth baselines are always evaluated as well.
        Selection is made over the combined Pareto front of whichever run, so
        naming a single solver is supported but gives up the pooling. The
        solvers' own hyperparameters (``n_samples``, ``seed``, ``maxfun``,
        ``n_scaling_factors``, ``round_type``, ``init_with_largest``) are not
        exposed; FIFO-Advisor's defaults are used.
    selection (str | callable):
        Which point of the front to apply. ``'min_bram'`` (default) is the rule
        validated against real RTL; ``'min_latency'`` takes the fastest point
        instead; a callable receives the front and returns one point.
    search_scope (str):
        ``'compute'`` (default) lets the solvers vary only the hls4ml
        inter-layer FIFOs -- exactly the channels this pass may write back. The
        AXI wrapper and batching channels keep their as-generated depths, stay
        simulated, and still count toward the reported cost, so the winner
        describes the design that will actually be built. ``'all'`` also lets
        the solvers assign depths to the wrapper channels; those assignments
        are discarded at writeback, so the reported cost then describes a
        design that is never built, and a candidate can appear to deadlock
        because of a wrapper depth that will not exist.
    write_report (bool):
        Write ``fifo_depths_advisor.json`` beside ``fifo_depths.json``, holding
        the winner (with the solver that found it), the Pareto front, per-solver
        bests and depths, channel counts and stage timings.
    skip_synthesis (bool):
        Analyse an already-synthesised solution instead of running C synthesis.
        Only correct if that solution was synthesised at the default depths.
    """

    def __init__(self):
        # Which FIFO-Advisor strategies to run. All of them by default: no
        # single strategy dominates across models, so the selection is made
        # over the combined Pareto front rather than any one solver's best.
        self.solvers = _KNOWN_SOLVERS
        # 'min_bram' (default), 'min_latency', or a callable taking the front.
        self.selection = 'min_bram'
        # 'compute' (default): the solvers may only vary the hls4ml
        # inter-layer FIFOs -- exactly the channels this pass is allowed to
        # write back. The AXI wrapper and batching channels keep their
        # as-generated depths, stay simulated, and still count toward the
        # reported latency and BRAM, so the winner's cost describes the design
        # that will actually be built. 'all' additionally lets the solvers
        # assign depths to the wrapper channels; those choices are discarded at
        # writeback, so the reported cost then describes a design that is never
        # built (and a candidate can appear to deadlock because of a wrapper
        # depth that will not exist).
        self.search_scope = 'compute'
        # Write fifo_depths_advisor.json next to the project for post-processing.
        self.write_report = True
        # Reuse an already-synthesised solution instead of running C synthesis.
        self.skip_synthesis = False

    def transform(self, model):
        """Size the inter-layer FIFOs by static analysis of the synthesised design.

        Args:
            model (ModelGraph): The model to which FIFO depth optimization is applied.

        Raises:
            RuntimeError: If the IO type is not "io_stream", if ``lightningsim``
                or ``fifo_advisor`` is not installed, if the helper modules
                cannot be imported, or if the reconstructed project descriptor
                disagrees with the backend .cfg it was derived from.
            ValueError: If ``search_scope``, ``selection`` or ``solvers`` names
                a value the pass does not recognise. Both this and the
                RuntimeErrors above are raised before any synthesis runs.

        Returns:
            bool: The execution state of the Optimizer Pass
        """
        # Validate the whole configuration BEFORE the expensive C synthesis, so
        # a typo costs seconds, not a synthesis run.
        if model.config.get_config_value('IOType') != 'io_stream':
            raise RuntimeError('To use this optimization you have to set `IOType` field to `io_stream` in the HLS config.')
        if self.search_scope not in ('all', 'compute'):
            raise ValueError(f"search_scope must be 'all' or 'compute', got {self.search_scope!r}")
        if not callable(self.selection) and self.selection not in ('min_bram', 'min_latency'):
            raise ValueError(f"selection must be 'min_bram', 'min_latency' or a callable, got {self.selection!r}")
        unknown_solvers = [s for s in self.solvers if s not in _KNOWN_SOLVERS]
        if unknown_solvers:
            raise ValueError(f'unknown solver(s) {unknown_solvers}; known: {list(_KNOWN_SOLVERS)}')

        # The analysis packages are optional and cannot be declared as hls4ml
        # dependencies, so check for them here -- before the C synthesis rather
        # than deep inside the sweep. Importing the helper modules below would
        # not catch their absence: they defer their own third-party imports, so
        # the import succeeds and the failure surfaces much later, after a
        # synthesis run has already been paid for. find_spec does not import.
        missing = [name for name in ('lightningsim', 'fifo_advisor') if importlib.util.find_spec(name) is None]
        if missing:
            raise RuntimeError(
                f'FIFO-Advisor depth optimization requires the {" and ".join(missing)} '
                f'package{"s" if len(missing) > 1 else ""}, which {"are" if len(missing) > 1 else "is"} '
                'not installed. Both need Python >= 3.11 on linux-64; LightningSim is distributed '
                'through the project conda channel (https://sharc-lab.github.io/LightningSim/repo), '
                'not PyPI. See docs/advanced/fifo_depth.rst for the environment file.'
            )

        try:
            from .fifo_advisor import hls_app, ls_compat
            from .fifo_advisor.sweep import analyze_solution
        except ImportError as exc:  # pragma: no cover - depends on the user's env
            raise RuntimeError(
                f'FIFO-Advisor depth optimization could not import its helper modules. Original error: {exc}'
            ) from exc

        # Must be installed before anything imports lightningsim.trace_file.
        ls_compat.install()

        output_dir = model.config.get_output_dir()
        project_name = model.config.get_project_name()

        # C synthesis at the *default* depths: no inflation, no co-simulation.
        synthesis_wall = 0.0
        if not self.skip_synthesis:
            t_synth = time.time()
            model.write()
            model.build(reset=False, csim=False, synth=True, cosim=False, bitfile=False, log_to_stdout=False)
            synthesis_wall = round(time.time() - t_synth, 3)

        exec_dir = model.config.backend.writer.get_vitis_hls_exec_dir(model)
        solution_dir = os.path.join(exec_dir, 'hls')
        cfg_path = os.path.join(output_dir, 'hls_kernel_config.cfg')
        hls_app_path = os.path.join(exec_dir, 'hls.app')

        # LightningSim reads the classic flow's project descriptor, which the
        # v++-driven backend never writes. Reconstruct it, then check the
        # product against its source before trusting it.
        hls_app.generate(cfg_path, hls_app_path, project_name)
        ok_top, ok_files, _ = hls_app.verify(cfg_path, hls_app_path)
        if not (ok_top and ok_files):
            raise RuntimeError('reconstructed hls.app does not match the backend .cfg it was derived from')

        # The hls4ml stream variables that can receive a depth. The wrapper and
        # batching channels (batch_size_*, gmem_*, stream_in0_*, stream_out0_*)
        # have no hls4ml output variable, and the model input and output
        # streams are implementation dependant -- the co-simulation pass
        # excludes them for the same reason.
        known = {v.name for v in model.output_vars.values() if 'StreamVariable' in str(type(v)) and v.pragma}

        result = analyze_solution(
            solution_dir,
            solvers=self.solvers,
            selection=self.selection,
            restrict_names=known if self.search_scope == 'compute' else None,
        )

        depths = {name: depth for name, depth in result.depths.items() if name in known}

        initial_depths = {
            v.name: int(v.pragma[1])
            for v in model.output_vars.values()
            if 'StreamVariable' in str(type(v)) and isinstance(v.pragma, tuple) and v.name in depths
        }

        # Apply the winner only if it strictly improves the modelled cost, or
        # ties it with fewer total slots. FIFO-Advisor's design space floors
        # every depth at 2, so on designs whose defaults are already minimal
        # (e.g. pure Dense/MLP networks, all depths 1) the "winner" ties the
        # baseline on the cost model while using MORE slots than the defaults;
        # writing it back would make the design strictly worse. Leaving the
        # defaults in place is the correct result there.
        baseline = result.per_solver.get('baseline', {})
        base_cost = (baseline.get('best_bram'), baseline.get('best_latency'))
        win_cost = (result.winner['bram'], result.winner['latency'])
        strictly_better = None not in base_cost and win_cost < base_cost
        fewer_slots = sum(depths.values()) < sum(initial_depths.values())
        if strictly_better or fewer_slots:
            set_optimized_fifo_depths(model, depths)
        else:
            print(
                'FIFO-Advisor: default depths are already optimal '
                f'(winner {win_cost} does not beat baseline {base_cost} and uses '
                f'{sum(depths.values())} slots vs {sum(initial_depths.values())}); leaving them unchanged'
            )
            depths = dict(initial_depths)
        generate_depths_file(model, initial_depths, depths)

        if self.write_report:
            report = result.to_dict()
            report['search_scope'] = self.search_scope
            report['timings_s']['synthesis'] = synthesis_wall
            report['applied_depths'] = depths
            with open(os.path.join(output_dir, 'fifo_depths_advisor.json'), 'w') as report_file:
                json.dump(report, report_file, indent=2)

        print(
            f'FIFO optimization completed (FIFO-Advisor): applied {len(depths)} channel depths, '
            f'estimated latency {result.winner["latency"]} / BRAM {result.winner["bram"]}'
        )

        return False
