#!/usr/bin/env python3
"""Shared dataset vocabulary for the bridge connectome extractors.

One home for the pinned upstream-annotation vocabularies that map lab
labels onto simulation concepts: functional class, side, status filtering,
transmitter sign. extract.py (MaleCNS dialect) and extract_flywire.py
(FlyWire dialect) both import from here instead of pinning private copies,
so the two datasets cannot drift apart silently.

Loud-unmapped contract: every classifier takes a VocabReport and records
any raw value that matched no vocabulary entry before falling through to
its default (inter / side 0 / row dropped). Unmapped values are echoed
with counts; run an extractor with --strict or VOCAB_STRICT=1 to turn any
unmapped value into a hard failure. Vocabulary drift in an upstream
release must force a conscious vocab.py update, never a silent
reclassification.

Value sets below were enumerated from the pinned v1.0 / v783 inputs on
2026-09-17; extending a set is a deliberate act documented in the commit
that extends it.
"""
from __future__ import annotations

import os
import sys
from collections import Counter

# Fixed-point scale for i16 CSR weights (kernel contract, SCALE = 512).
SCALE = 512

# Presynaptic transmitter -> signed multiplier. Modulators at +0.5 are an
# assumption (tunable); kept identical across both dataset dialects.
SIGN = {
    "acetylcholine": 1.0,
    "gaba": -1.0,
    "glutamate": -1.0,
    "histamine": -1.0,
    "dopamine": 0.5,
    "serotonin": 0.5,
    "octopamine": 0.5,
    "tyramine": 0.5,
}

# func_class -> u8 code stored in func_classes.u8 (kernel contract).
FUNC_CODE = {"inter": 0, "sensory": 1, "dn": 2, "motor": 3}


class VocabReport:
    """Accumulates raw values that fell through a classifier to a default."""

    def __init__(self, name: str):
        self.name = name
        self.falls: Counter = Counter()

    def record(self, value, n: int = 1) -> None:
        self.falls[value] += n

    def echo(self, strict: bool) -> bool:
        """Print the report; True if any unmapped values were seen."""
        if not self.falls:
            print(f"vocab[{self.name}]: all values recognized")
            return False
        total = sum(self.falls.values())
        print(f"vocab[{self.name}]: {total:,} rows hit UNMAPPED values:")
        for value, n in self.falls.most_common():
            print(f"  {value!r}: {n:,} rows")
        print("  -> classify these deliberately in bridge/vocab.py"
              + ("  [strict mode: FAILING]" if strict else ""))
        return True


def strict_mode() -> bool:
    return bool(os.environ.get("VOCAB_STRICT")) or "--strict" in sys.argv


def finish(reports, strict: bool | None = None) -> None:
    """Echo every report; exit(1) under strict mode if anything is unmapped."""
    strict = strict_mode() if strict is None else strict
    bad = any(rep.echo(strict) for rep in reports)
    if bad and strict:
        sys.exit(f"vocab: unmapped values present and VOCAB_STRICT=1")


# --------------------------------------------------------------------------
# MaleCNS v1.0 dialect (extract.py). Exact string matches, case-sensitive —
# that is the upstream vocabulary.
# --------------------------------------------------------------------------

# statusLabel values kept as simulated segments (positive curation).
KEEP_STATUS = frozenset({
    "Reviewed", "Roughly traced", "Prelim Roughly traced", "RT Hard to trace",
    "Hard to trace", "Partially traced", "0.5assign",
    "Anchor", "Soma Anchor", "Sensory Anchor",
})

# statusLabel values deliberately dropped (observed in v1.0, incl. 0-count).
EXCLUDE_STATUS = frozenset({
    "Orphan", "Glia", "Unimportant", "Out of scope", "Orphan-artifact",
    "Leaves", "Orphan hotknife", "PRT Orphan", "RT Orphan",
    "Putative Leaves", "Not examined", "", "Will be merged",
    "Cleaved Anchor", "Cervical Anchor", "Examined Soma Anchor",
    "Primary Anchor", "Traced in ROI", "Traced", "Finalized",
})

# superclass -> sensory
SEN = frozenset({
    "ol_sensory", "cb_sensory", "vnc_sensory", "cb_sensory_tbc",
    "vnc_sensory_tbc", "sensory_ascending", "sensory_ascending_tbc",
    "sensory_descending",
})
# superclass -> descending neuron
DN = frozenset({
    "descending_neuron", "descending_neuron_tbc", "efferent_descending",
})
# superclass -> motor
MN = frozenset({
    "vnc_motor", "cb_motor", "vnc_efferent", "cb_efferent", "vnc_tbc",
})
# Motor neurons are glutamatergic even when the predictor returns 'unclear'
# (they synapse peripherally, so the predictor has no signal).
MOTOR_DEFAULT_NT = frozenset({"vnc_motor", "cb_motor"})

# superclass -> interneuron, enumerated explicitly so a NEW superclass value
# can never silently join this set (it would land in VocabReport instead).
INTER = frozenset({
    "ol_intrinsic", "cb_intrinsic", "vnc_intrinsic", "visual_projection",
    "ascending_neuron", "visual_centrifugal", "cb_endocrine", "ENS",
    "vnc_endocrine", "efferent_ascending", "visual_projection_tbc",
})

# somaSide/rootSide -> side code (kernel contract: 0 unknown, 1 L, 2 R, 3 midline).
# v1.0 writes midline as "M" ("midline" also seen in derived tables).
MALECNS_SIDE = {"L": 1, "R": 2, "M": 3, "midline": 3}


def _isna(v) -> bool:
    return v is None or (isinstance(v, float) and v != v)


def malecns_status_classifier(rep: VocabReport):
    """True = keep the row. Unrecognized statuses drop AND get reported.

    NaN-first: missing status never matches a keep value, regardless of
    column dtype (categorical map can skip NaN without calling this)."""
    def keep(raw) -> bool:
        if _isna(raw):
            return False
        s = str(raw)
        if s in KEEP_STATUS:
            return True
        if s in EXCLUDE_STATUS:
            return False
        rep.record(s)
        return False
    return keep


def malecns_func_core(raw) -> tuple[str, bool]:
    """(func_class, known) — known=False means vocabulary fall-through."""
    sc = str(raw)
    if sc in SEN:
        return "sensory", True
    if sc in DN:
        return "dn", True
    if sc in MN:
        return "motor", True
    if sc in INTER:
        return "inter", True
    return "inter", False


def malecns_func_classifier(rep: VocabReport):
    def classify(raw) -> str:
        label, known = malecns_func_core(raw)
        if not known:
            rep.record(str(raw))
        return label
    return classify


def _fmt(v) -> str:
    if v is None:
        return "None"
    s = str(v)
    return "NaN" if s in ("nan", "<NA>") else s


def malecns_side_classifier(rep: VocabReport):
    """Replicates the somaSide->rootSide fallback chain per row."""
    def side(soma, root) -> int:
        for v in (soma, root):
            if v in MALECNS_SIDE:
                return MALECNS_SIDE[v]
        rep.record(f"somaSide={_fmt(soma)},rootSide={_fmt(root)}")
        return 0
    return side


# --------------------------------------------------------------------------
# FlyWire Fafb v783 dialect (extract_flywire.py). Values lowercased first.
# --------------------------------------------------------------------------

# TSV side column -> side code.
SIDE = {"left": 1, "right": 2, "center": 3}
# Known "no side" vocabulary (with missing/<NA>) -> 0, not a vocab miss.
SIDE_UNKNOWN = frozenset({"na"})

# super_class -> func_class. Explicit sets replacing the old substring rule
# ("sensory"/"visual" contained, or == "optic"); identical behavior on the
# pinned v783 vocabulary, but a new value is now loud instead of guessed.
FW_SENSORY = frozenset({
    "sensory", "optic", "visual_projection", "visual_centrifugal",
    "sensory_ascending",
})
FW_DN = frozenset({"descending"})
FW_MOTOR = frozenset({"motor"})
FW_INTER = frozenset({"central", "ascending", "endocrine"})

# status values deliberately dropped (this TSV flags problem segments rather
# than curating positives; keep-rule is status-is-missing).
FW_EXCLUDE_STATUS = frozenset({"outlier_seg", "outlier_bio"})


def flywire_status_classifier(rep: VocabReport):
    """True = keep (FlyWire keep-rule is status-is-missing). NaN-first so a
    categorical/NA dtype can never bypass the keep decision."""
    def keep(raw) -> bool:
        if _isna(raw):
            return True
        s = str(raw).lower()
        if s in FW_EXCLUDE_STATUS:
            return False
        rep.record(s)
        return False
    return keep


def flywire_func_core(raw) -> tuple[str, bool]:
    """(func_class, known) — known=False means vocabulary fall-through."""
    sc = str(raw).lower()
    if sc in FW_SENSORY:
        return "sensory", True
    if sc in FW_DN:
        return "dn", True
    if sc in FW_MOTOR:
        return "motor", True
    if sc in FW_INTER:
        return "inter", True
    return "inter", False


def flywire_func_classifier(rep: VocabReport):
    def classify(raw) -> str:
        label, known = flywire_func_core(raw)
        if not known:
            rep.record(str(raw))
        return label
    return classify


def flywire_side_classifier(rep: VocabReport):
    def side(raw) -> int:
        s = str(raw).lower()
        if s in SIDE:
            return SIDE[s]
        if s in SIDE_UNKNOWN or s in ("<na>", "nan", "none", ""):
            return 0
        rep.record(s)
        return 0
    return side
