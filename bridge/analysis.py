#!/usr/bin/env python3
"""Shared analysis helpers for the bridge battery scripts.

Single home for the functional-dimorphism primitives that
dimorphism_battery.py and dn_profile_battery.py previously carried as
private copies: the Fornito-style dimorphism index, the two-proportion
enrichment z, and the MaleCNS status keep-set (re-exported from vocab so
extractors and analysis cannot drift).
"""
from __future__ import annotations

import math

import numpy as np

from vocab import KEEP_STATUS  # re-export: analysis and extractors share one set

__all__ = ["KEEP_STATUS", "fdi", "enrichment_z", "top_quartile_mask"]


def fdi(m, f):
    """Functional dimorphism index: ||m - f|| / (||m|| + ||f||) in [0, 1]."""
    dm, df = np.asarray(m, float), np.asarray(f, float)
    return float(np.linalg.norm(dm - df) /
                 (np.linalg.norm(dm) + np.linalg.norm(df) + 1e-9))


def top_quartile_mask(values):
    """Boolean mask of values >= their own 75th percentile."""
    arr = np.asarray(values, float)
    return arr >= np.quantile(arr, 0.75)


def enrichment_z(fdi_values, flagged, label=""):
    """Two-proportion z: are flagged types over-represented in the top-FDI
    quartile? Returns (z, stats) with the 2x2 cell counts."""
    hi = top_quartile_mask(fdi_values)
    mask = np.asarray(flagged, bool)
    a = int((hi & mask).sum())
    b = int((hi & ~mask).sum())
    c = int((~hi & mask).sum())
    d = int((~hi & ~mask).sum())
    p_hi = a / max(a + b, 1)
    p_lo = c / max(c + d, 1)
    p_pool = (a + c) / max(a + b + c + d, 1)
    se = math.sqrt(p_pool * (1 - p_pool) *
                   (1 / max(a + b, 1) + 1 / max(c + d, 1))) or 1e-9
    z = (p_hi - p_lo) / se
    stats = {"a": a, "b": b, "c": c, "d": d, "p_hi": p_hi, "p_lo": p_lo}
    if label:
        print(f"{label:<24} top-FDI flagged: {a}/{a+b} ({p_hi:.0%}) vs "
              f"{c}/{c+d} ({p_lo:.0%})  z={z:+.2f}")
    return z, stats
