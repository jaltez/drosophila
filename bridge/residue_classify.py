#!/usr/bin/env python3
"""TypeSafe residue classifier (pilot for the extraction-vocabulary work).

The exact-match vocabulary in bridge/vocab.py is the fast path mapping
upstream annotation rows onto functional classes. Rows it cannot classify
deliberately (superclass values absent from the pinned vocabulary) are the
residue: this script asks one Jev Choice question per residue row and
routes by confidence (classification_using_confidence shape) —

  choice != 'review' and confidence >= CONFIDENT   -> accepted
  otherwise                                        -> review queue

The model only ever picks among the four functional classes or explicit
escalation; code owns the artifact. Nothing here runs inside the
simulation: this is an offline, once-per-dataset-release tool.

Modes:
  (default)                  scan both datasets for residue and classify it
  --dataset malecns|flywire  restrict to one dataset
  --validate N               blind-ask N stratified rows the exact path DOES
                             classify, and score agreement (calibration on
                             known answers)
  --all                      classify every kept row (dataset migration use;
                             expensive)

Output: bridge/out/residue_classified.json
"""
from __future__ import annotations

import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import pyarrow.feather as feather
from typesafe_sdk import Choice

import vocab
from tsafe import AnswerCache, cache_key, get_client

DATA = Path(__file__).parent / "data"
OUT = Path(__file__).parent / "out" / "residue_classified.json"
CONFIDENT = 0.8
NAMESPACE = "residue_func_class"

# Per-dataset criteria gloss: each dataset's own exact-match semantics,
# spelled out so the judgment mirrors (not overrides) the vocabulary.
CRITERIA = {
    "malecns": {
        "sensory": "Sensory afferent: peripheral sense origin (olfactory, "
                   "visual, other), including optic-lobe sensory, cb/vnc "
                   "sensory, sensory_ascending, sensory_descending.",
        "dn": "Descending neuron: axon leaves the brain toward the VNC to "
              "drive behavior.",
        "motor": "Motor/efferent neuron to muscle (VNC or cranial).",
        "inter": "Intrinsic or other: optic-lobe/cb/vnc intrinsic, ascending "
                 "interneurons, endocrine, ENS, visual_projection, or "
                 "anything not clearly afferent, efferent, or descending.",
        "review": "Too ambiguous to place from this row - escalate to a "
                  "human curator instead of guessing.",
    },
    "flywire": {
        "sensory": "Sensory or visual: peripheral sense origin, or optic-lobe "
                   "visual neurons (optic, visual_projection, "
                   "visual_centrifugal), sensory, sensory_ascending.",
        "dn": "Descending neuron (super_class 'descending').",
        "motor": "Motor neuron (super_class 'motor').",
        "inter": "Central intrinsic/interneuron: 'central', 'ascending', "
                 "'endocrine', or anything not clearly sensory/visual, "
                 "descending, or motor.",
        "review": "Too ambiguous to place from this row - escalate to a "
                  "human curator instead of guessing.",
    },
}

INSTRUCTIONS = (
    "One neuron's annotation row from the {dataset} Drosophila connectome "
    "dataset. The row's coarse label `super_class` is NOT in this project's "
    "pinned vocabulary, so judge the neuron's functional class from the "
    "biological meaning of all the fields. Which functional class is it?"
)


def load_rows(dataset: str) -> pd.DataFrame:
    if dataset == "malecns":
        an = feather.read_table(
            DATA / "body-annotations-male-cns-v1.0-minconf-0.5.feather",
            columns=["bodyId", "statusLabel", "superclass", "type",
                     "instance", "entryNerve", "somaSide", "rootSide"],
        ).to_pandas()
        keep = an["statusLabel"].isin(vocab.KEEP_STATUS) & an["superclass"].notna()
        an = an[keep].copy()
        an = an.rename(columns={
            "statusLabel": "status", "superclass": "super_class",
            "entryNerve": "entry_nerve", "somaSide": "soma_side",
            "rootSide": "root_side", "bodyId": "id"})
    elif dataset == "flywire":
        an = pd.read_csv(
            DATA / "Supplemental_file1_neuron_annotations.tsv", sep="\t",
            low_memory=False,
            usecols=["root_id", "status", "super_class", "cell_type",
                     "cell_class", "side", "nerve"])
        an = an[an["status"].isna() & an["super_class"].notna()].copy()
        an = an.rename(columns={"root_id": "id"})
    else:
        sys.exit(f"unknown dataset {dataset!r}")
    return an.reset_index(drop=True)


def row_state(dataset: str, row) -> dict:
    fields = ["super_class", "cell_type", "cell_class", "side", "status",
              "type", "instance", "entry_nerve", "soma_side", "root_side"]
    state = {"dataset": dataset}
    for f in fields:
        if f in row.index:
            v = row[f]
            state[f] = None if pd.isna(v) else (v.item() if hasattr(v, "item") else v)
    return state


def residue_ids(dataset: str, rows: pd.DataFrame) -> list[int]:
    core = (vocab.malecns_func_core if dataset == "malecns"
            else vocab.flywire_func_core)
    return [i for i, raw in enumerate(rows["super_class"])
            if not core(raw)[1]]


def ask(client, cache, dataset: str, state: dict) -> dict:
    key = cache_key(NAMESPACE, state)
    hit = cache.get(key)
    if hit is not None:
        return hit
    if client is None:
        raise RuntimeError(
            f"offline: TYPESAFE_API_KEY unset and cache miss for "
            f"{key[:12]}... — set the key or ship the cache entry")
    resp = client.system_one(
        state=state,
        questions={"func_class": Choice(
            instructions=INSTRUCTIONS.format(dataset=dataset),
            criteria=CRITERIA[dataset])},
    )
    a = resp.answers["func_class"]
    out = {"choice": a.choice, "confidence": a.confidence,
           "probabilities": dict(a.probabilities or {})}
    cache.put(key, out)
    return out


def route(answer: dict) -> str:
    if answer["choice"] == "review" or answer["confidence"] < CONFIDENT:
        return "review"
    return "accepted"


def main() -> None:
    datasets = ["malecns", "flywire"]
    argv = sys.argv[1:]
    if "--dataset" in argv:
        datasets = [argv[argv.index("--dataset") + 1]]
    validate_n = int(argv[argv.index("--validate") + 1]) if "--validate" in argv else 0
    classify_all = "--all" in argv

    client = get_client()
    cache = AnswerCache()
    results = {"accepted": {}, "review": {}, "validation": {}}

    for dataset in datasets:
        rows = load_rows(dataset)
        core = (vocab.malecns_func_core if dataset == "malecns"
                else vocab.flywire_func_core)
        exact = [core(raw)[0] for raw in rows["super_class"]]

        if validate_n:
            rng = random.Random(20260917)
            by_class: dict[str, list[int]] = {}
            for i, lab in enumerate(exact):
                by_class.setdefault(lab, []).append(i)
            per = max(1, validate_n // len(by_class))
            sample = sorted(i for lab, idxs in by_class.items()
                            for i in rng.sample(idxs, min(per, len(idxs))))
            with ThreadPoolExecutor(max_workers=6) as pool:
                answers = list(pool.map(
                    lambda i: ask(client, cache, dataset, row_state(dataset, rows.iloc[i])),
                    sample))
            stats: dict[str, dict] = {}
            for i, ans in zip(sample, answers):
                lab = exact[i]
                s = stats.setdefault(lab, {"n": 0, "agree": 0, "review": 0,
                                           "disagree": [], "reviewed": []})
                s["n"] += 1
                if ans["choice"] == "review" or ans["confidence"] < CONFIDENT:
                    s["review"] += 1
                    s["reviewed"].append(
                        {"id": str(rows.iloc[i]["id"]),
                         "super_class": str(rows.iloc[i]["super_class"]),
                         "confidence": ans["confidence"]})
                elif ans["choice"] == lab:
                    s["agree"] += 1
                else:
                    s["disagree"].append(
                        {"id": str(rows.iloc[i]["id"]),
                         "super_class": str(rows.iloc[i]["super_class"]),
                         "model": ans["choice"],
                         "confidence": ans["confidence"]})
            results["validation"][dataset] = stats
            print(f"[{dataset}] blind validation on {len(sample)} exact-classified rows:")
            for lab in sorted(stats):
                s = stats[lab]
                print(f"  {lab:<8} n={s['n']:<3} agree={s['agree']:<3} "
                      f"review-escalated={s['review']:<2} "
                      f"disagree={len(s['disagree'])}")
                for ex in s.get("reviewed", [])[:3]:
                    print(f"      review: {ex['super_class']!r} (conf {ex['confidence']:.2f})")
        else:
            ids = list(range(len(rows))) if classify_all else residue_ids(dataset, rows)
            print(f"[{dataset}] residue rows: {len(ids)}"
                  if not classify_all else f"[{dataset}] classifying all {len(ids)} rows")
            with ThreadPoolExecutor(max_workers=6) as pool:
                answers = list(pool.map(
                    lambda i: ask(client, cache, dataset, row_state(dataset, rows.iloc[i])),
                    ids))
            for i, ans in zip(ids, answers):
                bucket = route(ans)
                results[bucket].setdefault(dataset, {})[str(rows.iloc[i]["id"])] = ans

    cache.save()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
