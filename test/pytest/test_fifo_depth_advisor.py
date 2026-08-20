"""Tests for the FIFO-Advisor static FIFO-depth optimization pass.

The unit tests need neither Vitis nor the analysis packages: they cover the
pure-Python helpers (hls.app reconstruction, trace deduplication, the
LightningSim source patch) and the pass registration, and run in well under a
second.

The synthesis tests at the end mirror ``test_fifo_depth.py`` with the
co-simulation flow replaced by the FIFO-Advisor flow: same shape (build the
model, run the flow, then rebuild with the chosen depths and co-simulate to
prove no deadlock was introduced), and skipped by default exactly as the
existing FIFO-depth synthesis tests are.
"""

import json
import os
import re
from pathlib import Path

import numpy as np
import pytest
from tensorflow.keras.layers import Activation, Conv2D, Dense, Flatten, Input, MaxPooling2D
from tensorflow.keras.models import Model

import hls4ml
from hls4ml.backends.vitis_unified.passes.fifo_advisor import dedup, hls_app, ls_compat

test_root_path = Path(__file__).parent

ADVISOR_FLOW = 'vitisunified:fifo_depth_optimization_advisor'

# test_fifo_depth.py parametrizes over its backends the same way. The advisor
# flow is registered on VitisUnified only, so the list has one entry today --
# kept as a list so a second backend is a one-line change rather than a rewrite.
backend_options = ['VitisUnified']


# --- hls_app: reconstructing the classic project descriptor -----------------


def make_cfg(tmp_path):
    cfg = tmp_path / 'hls_kernel_config.cfg'
    cfg.write_text(
        '\n'.join(
            [
                'part=xczu9eg-ffvb1156-2-e',
                '[hls]',
                'syn.top=myproject_axi_master',
                f'syn.file={tmp_path}/firmware/myproject.cpp',
                f'syn.file_cflags={tmp_path}/firmware/myproject.cpp,-std=c++14',
                f'syn.file={tmp_path}/firmware/myproject_axi_master.cpp',
                f'tb.file={tmp_path}/myproject_test.cpp',
                f'tb.file_cflags={tmp_path}/myproject_test.cpp,-std=c++14',
                f'tb.file={tmp_path}/firmware/weights',
                'syn.cflags=-I.',
            ]
        )
    )
    return cfg


def test_hls_app_roundtrip(tmp_path):
    cfg = make_cfg(tmp_path)
    app = tmp_path / 'hls.app'
    hls_app.generate(cfg, app, 'myproject')
    ok_top, ok_files, n_files = hls_app.verify(cfg, app)
    assert ok_top and ok_files and n_files == 4


def test_hls_app_accepts_str_paths(tmp_path):
    # hls4ml's own code builds paths with os.path, so str must work everywhere.
    cfg = make_cfg(tmp_path)
    app = tmp_path / 'hls.app'
    hls_app.generate(str(cfg), str(app), 'myproject')
    assert hls_app.verify(str(cfg), str(app))[:2] == (True, True)


def test_hls_app_requires_top(tmp_path):
    cfg = tmp_path / 'empty.cfg'
    cfg.write_text('[hls]\n')
    with pytest.raises(RuntimeError, match='no syn.top'):
        hls_app.generate(cfg, tmp_path / 'hls.app', 'myproject')


# --- dedup: reducing the per-invocation FIFO list ---------------------------


class FakeFifo:
    def __init__(self, fifo_id, name):
        self.id = fifo_id
        self.name = name
        self.width = 16


class FakeEnv:
    def __init__(self, names, invocations):
        self.fifos = [FakeFifo(i, name) for i, name in enumerate(names * invocations)]
        self.num_fifos = len(self.fifos)


def test_dedup_keeps_first_instance_of_each_name():
    env = FakeEnv(['a', 'b', 'c'], invocations=5)
    env, stats = dedup.dedup_env(env, verify_compiled=False)
    assert stats == {'total': 15, 'distinct': 3, 'invocations': 5}
    assert [f.id for f in env.fifos] == [0, 1, 2]
    assert env.num_fifos == 3


def test_dedup_refuses_nonuniform_multiplicity():
    env = FakeEnv(['a', 'b'], invocations=1)
    env.fifos.append(FakeFifo(99, 'a'))
    env.num_fifos = 3
    with pytest.raises(dedup.DedupError, match='do not repeat uniformly'):
        dedup.dedup_env(env, verify_compiled=False)


def test_dedup_refuses_empty():
    with pytest.raises(dedup.DedupError, match='no FIFOs'):
        dedup.dedup_env(FakeEnv([], invocations=1), verify_compiled=False)


def test_restrict_env_limits_search_space():
    env = FakeEnv(['layer2_out', 'layer3_out', 'batch_size_c'], invocations=1)
    env, n_kept = dedup.restrict_env(env, {'layer2_out', 'layer3_out'})
    assert n_kept == 2
    assert {f.name for f in env.fifos} == {'layer2_out', 'layer3_out'}


def test_restrict_env_refuses_no_overlap():
    env = FakeEnv(['a'], invocations=1)
    with pytest.raises(dedup.DedupError, match='left no channels'):
        dedup.restrict_env(env, {'zzz'})


# --- ls_compat: the LightningSim payload patch ------------------------------

STOCK_SHAPED_SOURCE = (
    'from .model import BasicBlock, CDFGRegion, Instruction, Function, Solution\n'
    'def resolve_trace():\n'
    '    def do_sync_work_batch():\n'
    '        if True:\n'
    '            if True:\n'
    '                if True:\n'
    '                    if True:\n'
    '                        if True:\n'
    '                            payload = event_instruction.operands[-1]\n'
    '                            assert payload is not None\n'
    '                            source_instruction = payload.source\n'
    '                            assert isinstance(source_instruction, Instruction)\n'
    '                            fifo_widths[entry.metadata.fifo.id] = (\n'
    '                                source_instruction.bitwidth\n'
    '                            )\n'
)


def test_ls_compat_patch_applies_to_stock_shaped_source():
    patched = ls_compat.patch_source(STOCK_SHAPED_SOURCE)
    assert 'from .model.cdfg_edge import CDFGEdge' in patched
    assert 'LS_COMPAT_STATS' in patched
    assert 'payload = event_instruction.operands[-1]\n' not in patched


def test_ls_compat_refuses_changed_source():
    # A LightningSim whose payload selection moved or multiplied must not be
    # silently mis-patched.
    with pytest.raises(RuntimeError, match='re-derive the patch'):
        ls_compat.patch_source(STOCK_SHAPED_SOURCE.replace('operands[-1]', 'operands[0]'))
    with pytest.raises(RuntimeError, match='re-derive the patch'):
        ls_compat.patch_source(STOCK_SHAPED_SOURCE + STOCK_SHAPED_SOURCE)


# --- registration -----------------------------------------------------------


def test_advisor_pass_registered_alongside_cosim_pass():
    hls4ml.backends.get_backend('VitisUnified')
    from hls4ml.model.flow import get_flow
    from hls4ml.model.optimizer import get_optimizer

    opt = get_optimizer(ADVISOR_FLOW)
    assert type(opt).__name__ == 'FifoDepthOptimizationAdvisor'
    assert opt.search_scope == 'compute'

    flow = get_flow(ADVISOR_FLOW)
    assert flow.optimizers[0] == ADVISOR_FLOW
    # the co-simulation pass is untouched and still registered
    assert get_flow('vitisunified:fifo_depth_optimization').name


def test_advisor_pass_rejects_io_parallel():
    hls4ml.backends.get_backend('VitisUnified')
    from hls4ml.model.optimizer import get_optimizer

    class StubConfig:
        def get_config_value(self, key):
            return 'io_parallel'

    class StubModel:
        config = StubConfig()

    with pytest.raises(RuntimeError, match=re.escape('`io_stream`')):
        get_optimizer(ADVISOR_FLOW).transform(StubModel())


def test_advisor_pass_rejects_unknown_scope():
    hls4ml.backends.get_backend('VitisUnified')
    from hls4ml.model.optimizer import get_optimizer

    class StubConfig:
        def get_config_value(self, key):
            return 'io_stream'

    class StubModel:
        config = StubConfig()

    opt = get_optimizer(ADVISOR_FLOW)
    opt.configure(search_scope='everything')
    try:
        with pytest.raises(ValueError, match='search_scope'):
            opt.transform(StubModel())
    finally:
        opt.configure(search_scope='compute')


def test_advisor_pass_validates_before_synthesis():
    # A typo in the configuration must fail fast, not after a synthesis run:
    # transform() validates selection and solver names before building anything.
    hls4ml.backends.get_backend('VitisUnified')
    from hls4ml.model.optimizer import get_optimizer

    class StubConfig:
        def get_config_value(self, key):
            return 'io_stream'

    class StubModel:
        config = StubConfig()

    opt = get_optimizer(ADVISOR_FLOW)
    try:
        opt.configure(selection='fastest')
        with pytest.raises(ValueError, match='selection'):
            opt.transform(StubModel())
        opt.configure(selection='min_bram', solvers=('heuristic', 'genetic'))
        with pytest.raises(ValueError, match='unknown solver'):
            opt.transform(StubModel())
    finally:
        opt.configure(selection='min_bram', solvers=('heuristic', 'sa', 'group-sa', 'random', 'group-random'))


def test_advisor_pass_reports_missing_analysis_packages(monkeypatch):
    # The packages cannot be declared as hls4ml dependencies, so the pass has to
    # say so itself -- and say it before the C synthesis, not from inside the
    # sweep afterwards. The helper modules import fine without them (they defer
    # their own third-party imports), so presence is checked with find_spec.
    import importlib.util

    import hls4ml.backends.vitis_unified.passes.fifo_depth_optimization_advisor as advisor_mod

    hls4ml.backends.get_backend('VitisUnified')
    from hls4ml.model.optimizer import get_optimizer

    class StubConfig:
        def get_config_value(self, key):
            return 'io_stream'

    class StubModel:
        config = StubConfig()

    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name in ('lightningsim', 'fifo_advisor'):
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(advisor_mod.importlib.util, 'find_spec', fake_find_spec)

    opt = get_optimizer(ADVISOR_FLOW)
    with pytest.raises(RuntimeError, match='lightningsim and fifo_advisor'):
        opt.transform(StubModel())


# --- the full flow (needs Vitis + lightningsim + fifo_advisor) --------------
#
# These mirror test_fifo_depth.py with the co-simulation flow replaced by the
# FIFO-Advisor flow, and are skipped by default for the same reason.


def build_tiny_cnn():
    """A small streaming CNN: convolution, activation, pooling, dense head.

    Small enough to synthesise quickly, but with real feature-map FIFOs whose
    conservative default depths the optimizer can reclaim.
    """
    x_in = Input(shape=(8, 8, 1), name='in')
    x = Conv2D(2, (3, 3), name='conv')(x_in)
    x = Activation('relu', name='act')(x)
    x = MaxPooling2D((2, 2), name='pool')(x)
    x = Flatten(name='flat')(x)
    x = Dense(3, name='dense')(x)
    return Model(inputs=x_in, outputs=x, name='tiny_cnn')


def build_skip_cnn():
    """A streaming CNN with a residual skip connection.

    The reconvergent path is what makes FIFO sizing a correctness question and
    not only a resource one: one branch must buffer while the other drains.
    """
    from tensorflow.keras.layers import Add

    x_in = Input(shape=(8, 8, 1), name='in')
    x = Conv2D(4, (3, 3), padding='same', name='conv1')(x_in)
    x = Activation('relu', name='act1')(x)
    skip = x
    x = Conv2D(4, (3, 3), padding='same', name='conv2')(x)
    x = Add(name='skip_add')([skip, x])
    x = Activation('relu', name='act2')(x)
    x = Flatten(name='flat')(x)
    x = Dense(3, name='dense')(x)
    return Model(inputs=x_in, outputs=x, name='skip_cnn')


def run_fifo_advisor_optimization_keras(
    model, output_dir, backend='VitisUnified', selection='min_bram', search_scope='compute'
):
    """Convert one Keras model through the FIFO-Advisor flow and return it."""
    config = hls4ml.utils.config_from_keras_model(model, granularity='name')
    config['Flows'] = [ADVISOR_FLOW]
    hls4ml.model.optimizer.get_optimizer(ADVISOR_FLOW).configure(selection=selection, search_scope=search_scope)

    return hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir=output_dir,
        project_name='myproject',
        backend=backend,
        board='zcu102',
        part='xczu9eg-ffvb1156-2-e',
        clock_period=5,
        io_type='io_stream',
        axi_mode='axi_master',
    )


def fifo_advisor_optimization_checks(hls_model):
    """The depths must improve on the defaults and survive real RTL."""
    output_dir = hls_model.config.get_output_dir()

    # the pass writes the same interface file as the co-simulation flow ...
    with open(output_dir + '/fifo_depths.json') as fifo_depths_file:
        fifo_depths = json.load(fifo_depths_file)
    assert fifo_depths, 'the pass applied no depths'
    assert all(fifo['optimized'] <= fifo['initial'] for fifo in fifo_depths.values())
    assert any(fifo['optimized'] < fifo['initial'] for fifo in fifo_depths.values())

    # ... plus its own audit report
    with open(output_dir + '/fifo_depths_advisor.json') as report_file:
        report = json.load(report_file)
    assert report['pareto_front'], 'no non-deadlocking assignment was found'
    assert report['winner']['latency'] is not None

    # rebuild at the chosen depths and co-simulate: a deadlock introduced by the
    # new depths would fail or hang here
    hls_model.build(reset=False, csim=False, synth=True, cosim=True, bitfile=False)

    exec_dir = hls_model.config.backend.writer.get_vitis_hls_exec_dir(hls_model)
    cosim_report = Path(exec_dir) / 'reports' / 'hls_cosim.rpt'
    assert os.path.isfile(cosim_report), 'co-simulation report not found'
    assert any('Pass' in line for line in cosim_report.read_text().splitlines())


@pytest.mark.skip(reason='Skipping synthesis tests for now')
@pytest.mark.parametrize('backend', backend_options)
def test_successful_execution_of_tiny_cnn(test_case_id, backend):
    """The optimizer reclaims the conservative depths of a small streaming CNN."""
    hls_model = run_fifo_advisor_optimization_keras(build_tiny_cnn(), str(test_root_path / test_case_id), backend=backend)
    fifo_advisor_optimization_checks(hls_model)


@pytest.mark.skip(reason='Skipping synthesis tests for now')
@pytest.mark.parametrize('backend', backend_options)
def test_successful_execution_of_skip_cnn(test_case_id, backend):
    """A residual skip connection is the case where depths must be correct."""
    hls_model = run_fifo_advisor_optimization_keras(build_skip_cnn(), str(test_root_path / test_case_id), backend=backend)
    fifo_advisor_optimization_checks(hls_model)


@pytest.mark.skip(reason='Skipping synthesis tests for now')
@pytest.mark.parametrize('backend', backend_options)
def test_runtime_error_on_io_parallel(test_case_id, backend):
    """io_parallel has no FIFOs to size: the pass must refuse it."""
    message = 'To use this optimization you have to set `IOType` field to `io_stream` in the HLS config.'
    model = build_tiny_cnn()
    config = hls4ml.utils.config_from_keras_model(model, granularity='name')
    config['Flows'] = [ADVISOR_FLOW]
    with pytest.raises(RuntimeError, match=re.escape(message)):
        hls4ml.converters.convert_from_keras_model(
            model,
            hls_config=config,
            output_dir=str(test_root_path / test_case_id),
            project_name='myproject',
            backend=backend,
            part='xczu9eg-ffvb1156-2-e',
            clock_period=5,
            io_type='io_parallel',
            axi_mode='axi_master',
        )


@pytest.mark.skip(reason='Skipping synthesis tests for now')
@pytest.mark.parametrize('backend', backend_options)
def test_predictions_match_after_optimization(test_case_id, backend):
    """Sizing FIFOs must not change what the design computes."""
    model = build_tiny_cnn()
    x = np.random.rand(5, 8, 8, 1).astype(np.float32)
    keras_prediction = model.predict(x)

    hls_model = run_fifo_advisor_optimization_keras(model, str(test_root_path / test_case_id), backend=backend)
    hls_model.compile()
    hls_prediction = hls_model.predict(np.ascontiguousarray(x)).reshape(keras_prediction.shape)

    np.testing.assert_allclose(hls_prediction, keras_prediction, rtol=0, atol=0.05)
