"""Helpers for the FIFO-Advisor static FIFO-depth optimization pass.

This sub-package supports ``vitisunified:fifo_depth_optimization_advisor``. It
lives in a directory (not a ``.py`` file) deliberately: the optimizer-pass
scanner in ``hls4ml.model.optimizer`` only registers top-level ``.py`` modules
in a backend's ``passes/`` directory, so nothing in here is mistaken for a
pass.

Modules:

- ``hls_app``  -- reconstruct the classic ``hls.app`` project descriptor that
  LightningSim needs from the ``.cfg`` the VitisUnified backend drives v++ with.
- ``dedup``    -- reduce LightningSim's per-invocation FIFO list to the
  physically distinct channels (hls4ml's wrapper calls the kernel once per
  test sample).
- ``ls_compat`` -- in-memory correction of LightningSim's ``fifo_write``
  payload-operand selection, which a Dense output layer trips.
- ``sweep``    -- run FIFO-Advisor's solvers over a synthesised solution and
  select an assignment from the combined Pareto front.

``lightningsim`` and ``fifo_advisor`` are imported lazily inside functions, so
importing hls4ml (and this package) works in environments that do not have
them.
"""
