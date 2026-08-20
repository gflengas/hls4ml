"""Correct LightningSim's fifo_write payload selection, in memory, at import time.

The defect
----------
``lightningsim/trace_file.py`` resolves the data operand of a ``fifo_write``
event as ``event_instruction.operands[-1]``. That holds for a layer whose
output word is produced by a single instruction. It does not hold for a layer
that packs several independently computed values into one wide stream word --
a ``bitconcatenate`` in the CDFG, which a Dense output layer produces -- where
the last operand is a scheduling/control edge instead. Trace resolution then
dies on ``assert isinstance(source_instruction, Instruction)``.

Why this is applied unconditionally
-----------------------------------
The replacement scans operands in reverse for the last one that is genuinely a
data (``CDFGEdge.INPUT``) edge resolving to a real ``Instruction``. Because
``reversed()`` starts at ``operands[-1]``:

* if the stock expression would have succeeded, ``operands[-1]`` satisfies the
  predicate and is returned on the first iteration -- an identical result;
* if the stock expression would have raised, the scan continues and either
  finds a real payload or yields ``None``, whereupon the original
  ``assert payload is not None`` fires exactly as before.

So the patched expression can only differ from the stock one on inputs where
the stock one raises. Evaluating the predicate on extra operands introduces no
new failure mode: ``CDFGEdge.source`` catches ``KeyError`` and returns ``None``
rather than raising, and ``and`` short-circuits so ``.source`` is only touched
on ``INPUT`` edges. There is therefore no model for which enabling this is
worse than leaving it off, and no need to predict from the hls4ml graph whether
a given network will trip the defect.

A second case the stock code cannot express: a write whose value is a *constant*
has no producing instruction at all, so no operand can resolve. hls4ml's
zero-padding layers (which ``padding='same'`` inserts) write literal zeros for
the border, and stock LightningSim dies on ``assert payload is not None``. Such
a write carries no width information, so the correction leaves that channel
unresolved and lets a later write to the same channel -- one carrying real data,
which does have a producer -- supply the width. The width lookup is already
guarded by ``fifo.id not in fifo_widths``, so no extra bookkeeping is needed.

``LS_COMPAT_STATS`` records how often the scan had to look past ``operands[-1]``
(``diverged``) and how often a write carried a constant payload
(``constant_payload``). A model that never trips either reports zero for both.

Mechanism
---------
``do_sync_work_batch`` is a closure inside ``resolve_trace``, so it cannot be
reached by rebinding a module attribute. The fix is applied to the module
source as it is loaded, via a ``sys.meta_path`` hook, so exactly one version of
``lightningsim.trace_file`` ever exists and no copy of the package is written
to disk. The installed ``lightningsim`` distribution is not modified. The
patch is pinned to the exact stock source text: if LightningSim changes, the
hook refuses to patch rather than patching wrongly.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys

MODULE = 'lightningsim.trace_file'

_STOCK_IMPORT = 'from .model import BasicBlock, CDFGRegion, Instruction, Function, Solution'

_STOCK_PAYLOAD = """                            payload = event_instruction.operands[-1]
                            assert payload is not None
                            source_instruction = payload.source
                            assert isinstance(source_instruction, Instruction)
                            fifo_widths[entry.metadata.fifo.id] = (
                                source_instruction.bitwidth
                            )"""

_FIXED_PAYLOAD = """                            _operands = event_instruction.operands
                            payload = None
                            for _index in range(len(_operands) - 1, -1, -1):
                                _operand = _operands[_index]
                                if (
                                    _operand is not None
                                    and _operand.type == CDFGEdge.INPUT
                                    and isinstance(_operand.source, Instruction)
                                ):
                                    payload = _operand
                                    if _index == len(_operands) - 1:
                                        LS_COMPAT_STATS["last_operand_ok"] += 1
                                    else:
                                        LS_COMPAT_STATS["diverged"] += 1
                                    break
                            LS_COMPAT_STATS["resolutions"] += 1
                            if payload is not None:
                                source_instruction = payload.source
                                assert isinstance(source_instruction, Instruction)
                                fifo_widths[entry.metadata.fifo.id] = (
                                    source_instruction.bitwidth
                                )
                            else:
                                # No operand of this write resolves to a producing
                                # instruction, which happens when the value written is a
                                # constant: hls4ml's zero-padding layers (inserted by
                                # padding='same') write literal zeros for the border.
                                # Such a write carries no width information, so this
                                # channel is left unresolved and a later write to it --
                                # one carrying real data, which does have a producer --
                                # supplies the width. The lookup is already guarded by
                                # `fifo.id not in fifo_widths`, so nothing else is needed.
                                LS_COMPAT_STATS["constant_payload"] += 1"""

_STATS_DECL = """

# Installed by hls4ml's fifo_advisor.ls_compat. "resolutions" counts payload
# lookups (one per FIFO id, the lookup is guarded by `fifo.id not in
# fifo_widths`); "diverged" counts the ones where operands[-1] was NOT the
# payload -- i.e. exactly the lookups where stock LightningSim would have
# raised.
LS_COMPAT_STATS = {"resolutions": 0, "last_operand_ok": 0, "diverged": 0, "constant_payload": 0}
"""


def patch_source(source):
    """Apply both edits to the text of trace_file.py, or explain why it cannot."""
    if source.count(_STOCK_PAYLOAD) != 1:
        raise RuntimeError(
            f'ls_compat: expected exactly one occurrence of the stock payload '
            f'selection in {MODULE}, found {source.count(_STOCK_PAYLOAD)}. '
            f'LightningSim has changed; re-derive the patch before trusting it.'
        )
    if source.count(_STOCK_IMPORT) != 1:
        raise RuntimeError(
            f'ls_compat: anchor import line not found exactly once in {MODULE}. '
            f'LightningSim has changed; re-derive the patch before trusting it.'
        )

    # CDFGEdge is NOT in trace_file's namespace; the import must be added or the
    # patched code dies with NameError (found the hard way in the IPFLOW
    # investigation).
    source = source.replace(
        _STOCK_IMPORT,
        _STOCK_IMPORT + '\nfrom .model.cdfg_edge import CDFGEdge' + _STATS_DECL,
        1,
    )
    return source.replace(_STOCK_PAYLOAD, _FIXED_PAYLOAD, 1)


class _PatchingLoader(importlib.machinery.SourceFileLoader):
    def get_data(self, path):
        data = super().get_data(path)
        if path.endswith('trace_file.py'):
            data = patch_source(data.decode()).encode()
        return data

    def get_code(self, fullname):
        # SourceFileLoader.get_code prefers a cached __pycache__/*.pyc and in
        # that case never reads the source, so get_data above would never run.
        # Compile from (patched) source unconditionally instead.
        path = self.get_filename(fullname)
        return self.source_to_code(self.get_data(path), path)


class _PatchingFinder:
    def find_spec(self, fullname, path=None, target=None):
        if fullname != MODULE:
            return None
        # Ask the rest of the meta path where the module lives, then load it
        # through our loader. Excluding ourselves avoids infinite recursion.
        finders = [f for f in sys.meta_path if f is not self]
        for finder in finders:
            find_spec = getattr(finder, 'find_spec', None)
            if find_spec is None:
                continue
            spec = find_spec(fullname, path, target)
            if spec is None or spec.origin is None:
                continue
            return importlib.util.spec_from_file_location(
                fullname, spec.origin, loader=_PatchingLoader(fullname, spec.origin)
            )
        return None


def install():
    """Install the hook. Must run before lightningsim.trace_file is imported."""
    if MODULE in sys.modules:
        raise RuntimeError(
            f'ls_compat.install() called after {MODULE} was already imported; '
            f'the patch would not apply to the loaded module. Import ls_compat '
            f'and call install() before importing lightningsim or fifo_advisor.'
        )
    if not any(isinstance(f, _PatchingFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _PatchingFinder())


def stats():
    """Payload-resolution counters, or an empty dict if nothing was resolved yet."""
    module = sys.modules.get(MODULE)
    if module is None:
        return {}
    return dict(getattr(module, 'LS_COMPAT_STATS', {}))


def verify_installed():
    """True if the loaded trace_file is the patched one."""
    module = sys.modules.get(MODULE)
    return module is not None and hasattr(module, 'LS_COMPAT_STATS')
