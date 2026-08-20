==============================
FIFO Buffer Depth Optimization
==============================

With the ``io_stream`` IO type, each layer is connected with the subsequent layer through first-in first-out (FIFO) buffers.
The implementation of the FIFO buffers contribute to the overall resource utilization of the design, impacting in particular the BRAM or LUT utilization.
Because the neural networks can have complex architectures generally, it is hard to know a priori the correct depth of each FIFO buffer.
By default ``hls4ml`` choses the most conservative possible depth for each FIFO buffer, which can result in a an unnecessary over-utilization of resources.

In order to reduce the impact on the resources used for FIFO buffer implementation, an optimization flow has been developed that correctly sizes the depth
of the FIFO buffers by analyzing the RTL co-simulation. This feature is currently available in ``Vitis`` and ``Vivado`` backends.

In ``Vivado`` backend, FIFO buffer resizing is implemented as a :py:class:`~hls4ml.backends.vivado.passes.fifo_depth_optimization` optimizer pass.
Through RTL simulation with large FIFO buffers (by default set to a depth of 100,000), we estimate the maximum occupation of each FIFO.
Once the maximum depth is determined, the optimizer pass sets the FIFO buffer depth to that value plus 1.

Below we show an example of the use of the FIFO depth optimization. First, we can define a simple neural network in Keras:

.. code-block:: Python

    from tensorflow.keras.layers import Dense
    from tensorflow.keras.models import Sequential

    model = Sequential()
    model.add(Dense(64, input_shape=(16,), name='fc1', activation='relu'))
    model.add(Dense(32, name='fc2', activation='relu'))
    model.add(Dense(32, name='fc3', activation='relu'))
    model.add(Dense(5, name='fc4', activation='softmax'))

Then, we can convert the model, including the flow:

.. code-block:: Python

    import hls4ml

    config = hls4ml.utils.config_from_keras_model(model, granularity='model')
    config['Flows'] = ['vivado:fifo_depth_optimization']
    hls4ml.model.optimizer.get_optimizer('vivado:fifo_depth_optimization').configure(profiling_fifo_depth=100_000)


    hls_model = hls4ml.converters.convert_from_keras_model(model,
                                                           io_type='io_stream',
                                                           hls_config=config,
                                                           output_dir='hls4mlprj_fifo_depth_opt',
                                                           part='xc7z020clg400-1',
                                                           backend='Vivado')

    hls_model.build(reset=False, csim=True, synth=True, cosim=True)

For more details and results, see `H. Borras et al., "Open-source FPGA-ML codesign for the MLPerf Tiny Benchmark" (2022) <https://arxiv.org/abs/2206.11791>`_.

Similarly, the FIFO buffers can be optimized while using the ``Vitis`` backend with the following changes:

.. code-block:: Python

    config['Flows'] = ['vitis:fifo_depth_optimization']
    hls4ml.model.optimizer.get_optimizer('vitis:fifo_depth_optimization').configure(profiling_fifo_depth=100_000)

    hls_model = hls4ml.converters.convert_from_keras_model(model,
                                                        io_type='io_stream',
                                                        hls_config=config,
                                                        output_dir='hls4mlprj_fifo_depth_opt',
                                                        part='xc7z020clg400-1',
                                                        backend='Vitis')

Static analysis with FIFO-Advisor (``VitisUnified`` backend)
============================================================

In the ``VitisUnified`` backend a second, static flow is available alongside the
co-simulation-based one: ``vitisunified:fifo_depth_optimization_advisor``. It
synthesises the design once at hls4ml's default depths, reconstructs a
cycle-accurate model of the schedule with `LightningSim
<https://github.com/sharc-lab/LightningSim>`_, and searches over depth
assignments with `FIFO-Advisor <https://github.com/sharc-lab/fifo-advisor>`_
instead of measuring occupancies in an RTL co-simulation. The search runs five
strategies and selects from their combined Pareto front; the chosen inter-layer
depths are then applied through the same writeback as the co-simulation flow,
so the two flows are interchangeable (enable exactly one of them).

The difference in what they compute: co-simulation records the occupancy that
naturally accumulates when FIFOs are effectively infinite, while the static
search finds the smallest depths that avoid a stall -- a tighter quantity.
The difference in cost: seconds of solver time after one synthesis at default
depths, against an RTL co-simulation of the design with every FIFO inflated to
100,000.

.. code-block:: Python

    config['Flows'] = ['vitisunified:fifo_depth_optimization_advisor']
    hls4ml.model.optimizer.get_optimizer('vitisunified:fifo_depth_optimization_advisor').configure(
        selection='min_bram',      # or 'min_latency', or a callable over the Pareto front
        search_scope='compute',    # 'all' also lets the solvers vary the wrapper channels
    )

    hls_model = hls4ml.converters.convert_from_keras_model(model,
                                                        io_type='io_stream',
                                                        hls_config=config,
                                                        output_dir='hls4mlprj_fifo_depth_advisor',
                                                        part='xczu9eg-ffvb1156-2-e',
                                                        backend='VitisUnified',
                                                        axi_mode='axi_master')

Next to ``fifo_depths.json`` (the same file the co-simulation flow writes), the
pass writes ``fifo_depths_advisor.json``:

``winner``
    the applied point -- latency, BRAM, its depths, and ``found_by``: every
    solver that reached it.
``pareto_front``
    each non-dominated point with its latency, BRAM, ``found_by`` and its own
    ``depths``, so the alternatives to the shipped assignment can be inspected.
``solvers``
    per solver: runtime, evaluation count, best latency/BRAM and ``best_depths``.
``fifo_stats`` / ``timings_s`` / ``ls_compat_stats``
    channel counts, per-stage wall times, and how often each LightningSim
    correction fired.

Individual evaluations are not recorded -- roughly 1800 are made across the six
strategies, and only the non-dominated ones are kept.

**Solver options.** ``solvers`` chooses which strategies run, but their own
hyperparameters are not exposed and FIFO-Advisor's defaults are used:
``n_samples=100`` and ``seed=7`` for the random searches, and ``maxfun=100``,
``n_scaling_factors=8``, ``round_type=RINT``, ``init_with_largest=False`` for
the annealing ones. FIFO-Advisor also implements strategies this flow does not
run -- a genetic optimizer, a group-exhaustive search, a continuous simulated
annealer, and multi-design variants. Anyone wanting to tune the search should
drive FIFO-Advisor directly against the synthesised solution; the pass writes
``hls.app`` beside the project, which is what its CLI needs.

Only the inter-layer FIFOs are ever written back, matching the co-simulation
flow's scope. ``search_scope`` controls what the solvers may *vary* while
searching: with the default ``'compute'`` the wrapper and batching channels
keep their as-generated depths (so the reported latency and BRAM describe the
design that will actually be built), while ``'all'`` also lets the solvers
assign depths to those channels, which are then discarded at writeback.

Installation
------------

Neither package is a dependency of hls4ml: both are imported lazily, only when
the flow actually runs, so hls4ml is unaffected if they are absent. To use the
flow they must be installed into the same environment as hls4ml, because the
pass calls the analysis in-process.

That environment has to be a conda environment, and this is not a preference.
LightningSim is not published on PyPI (``pip install lightningsim`` reports *no
matching distribution*), so a pip-only virtualenv cannot host the flow at all.
Its conda channel publishes **linux-64 only**, and it ships a compiled PyO3
extension built per interpreter version, with builds for **CPython 3.10, 3.11
and 3.12**. FIFO-Advisor requires Python >= 3.11. The intersection --
**linux-64, Python 3.11 or 3.12** -- is the flow's supported window; hls4ml
itself supports more than that, and only this flow is restricted.

Save this as ``environment.yml``:

.. code-block:: yaml

    name: hls4ml-fifo-advisor
    channels:
      - conda-forge
      - https://sharc-lab.github.io/LightningSim/repo
    dependencies:
      - python=3.11             # or 3.12; must match a published LightningSim build
      - lightningsim=0.2.6
      # FIFO-Advisor's runtime imports. It is installed with --no-deps below,
      # so these have to come from conda. scipy is pinned: conda-forge builds
      # 1.16+ against libstdc++ 15, and a pip-installed TensorFlow loads the
      # system libstdc++ first, so on a host whose system libstdc++ predates
      # GCC 13 the pair fails at import with `CXXABI_1.3.15 not found`. 1.15
      # is also the version every published measurement used.
      - numpy
      - scipy<1.16
      - pymoo
      # Declared by FIFO-Advisor but imported only by its experiments/ and
      # demo/ scripts. Listed so `pip check` stays clean; drop for a lean env.
      - pandas
      - matplotlib
      - seaborn
      # hls4ml's compiled dependencies, from conda rather than pip so that pip
      # does not overwrite the conda-provided numpy.
      - h5py
      - pyyaml
      - pip

then:

.. code-block:: bash

    conda env create -f environment.yml
    conda activate hls4ml-fifo-advisor

    # FIFO-Advisor, straight from git. --no-deps is what its own README
    # recommends: an unpinned resolve installs pip wheels on top of the
    # conda-provided numpy/scipy stack.
    pip install --no-deps git+https://github.com/sharc-lab/fifo-advisor.git

    # hls4ml itself (or `pip install -e .` from a checkout)
    pip install hls4ml

    # Add a model front end last, and only if you need one, e.g.
    #   pip install "hls4ml[qkeras]"     # TensorFlow 2.14 / Keras 2
    # This downgrades numpy to 1.26; the flow is unaffected, but pin the front
    # end rather than letting pip resolve it freely.

    # Verify
    python -c "import lightningsim, fifo_advisor, hls4ml; print('ok')"

Vitis must also be on ``PATH`` (``source /opt/Xilinx/Vitis/2023.2/settings64.sh``);
LightningSim invokes it to build the design's bitcode.

*Troubleshooting.* If you lift the ``scipy<1.16`` pin and install a Keras front
end, the pass may fail in the solver sweep with ``version 'CXXABI_1.3.15' not
found``: conda-forge's ``scipy`` >= 1.16 is built against libstdc++ 15, while a
pip-installed TensorFlow loads the *system* libstdc++ into the process first,
and on a host predating GCC 13 that one is too old. Either keep the pin or
export ``LD_LIBRARY_PATH=$CONDA_PREFIX/lib`` so conda's libstdc++ wins.
Environments without a Keras front end are unaffected either way.

Activating this environment exports ``CC``, ``CXX``, ``CFLAGS`` and ``LDFLAGS``
pointing at the conda toolchain. That is deliberate rather than incidental --
LightningSim reads ``CC``/``CXX`` from the environment to compile the traced
design into a native binary, which is why its conda package depends on
``gcc_linux-64``/``gxx_linux-64``. Vitis is unaffected by those variables.

**Adding the flow to an environment you already have.** If it is a conda
environment on Python 3.11 or 3.12, ``conda install -c conda-forge -c
https://sharc-lab.github.io/LightningSim/repo lightningsim=0.2.6 "scipy<1.16" pymoo``
works alongside a pip-installed hls4ml. Note that conda will take ownership of
``numpy`` if pip installed it; that is safe when the versions agree and is the
usual pip/conda hazard when they do not, so check ``pip check`` afterwards. If
your environment is a plain virtualenv, or is on Python 3.10, 3.13 or 3.14,
there is no way to add LightningSim to it and a separate conda environment is
the only route.

Requirements and limitations: the flow needs the ``lightningsim`` and
``fifo_advisor`` packages in the same Python environment as hls4ml, which
restricts it to linux-64 on Python 3.11 or 3.12 (see Installation above). It
currently supports the ``axi_master`` wrapper -- LightningSim cannot
link the testbench of an ``axi_stream`` top level (``hls::axis`` struct
streams), so for a design deployed behind AXI4-Stream the depths must be
derived on the ``axi_master`` twin of the same kernel. Depths proposed by the
static analysis are estimates of the schedule model; deadlock-freedom on real
RTL should be confirmed with one ordinary co-simulation of the rebuilt design.
