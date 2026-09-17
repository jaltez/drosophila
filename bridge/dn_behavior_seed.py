#!/usr/bin/env python3
"""DN type -> behavior vote seeding with a two-level Jev hierarchy.

DN type names (Namiki et al. 2018 taxonomy) carry a literature prior: the
class letter and known named types (MDN, the giant fiber DNp01, ...)
imply behavior roles. For each DN-shaped type NOT already curated in
scenarios/behavior_map.json, this script runs a hierarchical judgment
(hierarchical_classification shape, beam K=2):

  level 1 (family): Choice over behavior families
  level 2 (leaf):   Choice within the family (walk vs walk_back under
                    locomotion; 1:1 elsewhere - no extra request)

Path score = geometric mean of edge probabilities; separation = top/second
path. Confidence routing: leaf-equivalent confidence >= SEED_CONFIDENT
becomes a seed vote (weight stays null: weights are code/curator policy,
per composite scoring); anything else lands in the review section with
its family hint. Nothing here touches the curated map or the simulation
loop; output is a curation aid.

--validate blind-asks the curated types with name-only state (no evidence
strings) and scores agreement, split by whether the curated vote is
literature-grounded or came from this project's dn_screen measurements.

Output: scenarios/behavior_map_seed.json
"""
from __future__ import annotations

import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from typesafe_sdk import Choice

from tsafe import AnswerCache, cache_key, get_client

ROOT = Path(__file__).parent / ".."
ARTIFACT = ROOT / "bridge" / "out" / "connectome-malecns-v1.0" / "dn_types.json"
SCREEN = ROOT / "runs" / "dn_screen.json"
CURATED = ROOT / "scenarios" / "behavior_map.json"
OUT = ROOT / "scenarios" / "behavior_map_seed.json"

DN_SHAPED = re.compile(r"^(DNae|DNbe|DNge|DNpe|DNa|DNb|DNg|DNp|MDN)\d+$")
# Chosen on the curated-map validation: with screen state, judgments at
# confidence >= 0.9 agreed with curation 5/5 (all five screen-derived
# steer votes), while the sub-0.9 band agreed 1/5 (DNa02 at 0.63; the four
# misses sat at <= 0.71). Name-only types (no screen measurements) never
# seed: that evidence channel validated at 2/5 and is review-only.
SEED_CONFIDENT = 0.9
BEAM_K = 2
NAMESPACE_FAMILY = "dn_behavior_family"
NAMESPACE_LEAF = "dn_behavior_leaf"

FAMILIES = {
    "locomotion": ["walk", "walk_back"],
    "steering": ["steer"],
    "escape": ["escape"],
    "feeding": ["feed"],
    "grooming": ["groom"],
    "resting": ["rest"],
    "no-clear-role": [],
}

FAMILY_Q = Choice(
    instructions=(
        "You are judging one descending neuron (DN) type of the adult fruit "
        "fly Drosophila; type names follow the Namiki et al. 2018 DN "
        "taxonomy. When the state carries `unilateral_odor_screen`, those "
        "are this project's own connectome-simulation measurements: firing "
        "rate of the type under left-antenna-only (`left_hz`) vs "
        "right-antenna-only (`right_hz`) olfactory drive, with no-drive "
        "baseline `baseline_hz`. A strongly left/right-asymmetric response "
        "is the signature of a steering role; a symmetric rate increase "
        "fits locomotion. Weigh those measurements first where present, "
        "and the biology of the DN class/name second. Which behavior "
        "family does activating this type most support?"),
    criteria={
        "locomotion": "Forward or backward walking / locomotion drive "
                      "(walk-promoting DNs).",
        "steering": "Lateralized steering or turning control (left/right "
                    "asymmetric guidance of course).",
        "escape": "Escape: takeoff, jump, giant-fiber startle behaviors.",
        "feeding": "Feeding: proboscis extension / food intake drive.",
        "grooming": "Grooming: body-cleaning motor programs.",
        "resting": "Resting, stopping, quiescence.",
        "no-clear-role": "No well-supported behavior role is known for "
                         "this type name.",
    },
)

LEAF_Q = Choice(
    instructions=(
        "This Drosophila descending neuron type is judged locomotion-"
        "related. From its name and what is biologically known about it, "
        "which specific locomotion behavior fits best?"),
    criteria={
        "walk": "Forward walking.",
        "walk_back": "Backward walking (moonwalker-like).",
    },
)

# Curated votes fall into two evidence classes: those derived from this
# project's dn_screen measurements (the state carries the same numbers the
# curator saw) and those grounded in external literature (no screen state;
# the judgment rests on the name prior alone). The validation split below
# reports them separately.


def _load_screen() -> dict:
    if not SCREEN.exists():
        return {}
    rows = json.loads(SCREEN.read_text())
    out = {}
    for r in rows:
        out[r["type"]] = {
            "left_hz": round(r["left_hz"], 3),
            "right_hz": round(r["right_hz"], 3),
            "baseline_hz": round(r.get("base_hz", 0.0), 3),
            "n": r.get("n", 1),
        }
    return out


SCREEN_BY_TYPE = _load_screen()


def dn_state(name: str) -> dict:
    state = {"dn_type": name,
             "context": "descending neuron type, adult Drosophila (MaleCNS "
                        "connectome)"}
    if name in SCREEN_BY_TYPE:
        state["unilateral_odor_screen"] = SCREEN_BY_TYPE[name]
    return state


class Seeder:
    def __init__(self, client, cache: AnswerCache):
        self.client = client
        self.cache = cache

    def _ask(self, ns: str, question: Choice, state: dict) -> dict:
        key = cache_key(ns, state)
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        resp = self.client.system_one(state=state, questions={"q": question})
        a = resp.answers["q"]
        out = {"choice": a.choice, "confidence": a.confidence,
               "probabilities": dict(a.probabilities or {})}
        self.cache.put(key, out)
        return out

    def judge(self, name: str) -> dict:
        """Two-level hierarchy; returns leaf decision with path metrics."""
        state = dn_state(name)
        fam = self._ask(NAMESPACE_FAMILY, FAMILY_Q, state)
        probs = fam["probabilities"] or {fam["choice"]: fam["confidence"]}
        # (-value, name): model probabilities quantize coarsely, so exact
        # ties are common; dict order differs between a live response and a
        # JSON cache round-trip, so tie-break on the family name.
        top = sorted(probs.items(), key=lambda kv: (-kv[1], kv[0]))[:BEAM_K]

        paths = []  # (geo_mean, leaf, leaf_conf, family)
        for family, p in top:
            leaves = FAMILIES.get(family, [])
            if not leaves:
                paths.append((p, "no-vote", p, family))
            elif len(leaves) == 1:
                paths.append((p, leaves[0], p, family))
            else:
                leaf = self._ask(NAMESPACE_LEAF, LEAF_Q, state)
                lp = leaf["probabilities"] or {leaf["choice"]: leaf["confidence"]}
                conf = lp.get(leaf["choice"], leaf["confidence"])
                paths.append(((p * conf) ** 0.5, leaf["choice"], conf, family))
        paths.sort(key=lambda t: (-t[0], t[1]))
        geo, leaf, leaf_conf, family = paths[0]
        separation = (paths[0][0] / paths[1][0]) if len(paths) > 1 and paths[1][0] > 0 else float("inf")
        return {"behavior": leaf, "family": family,
                "confidence": leaf_conf, "path_score": geo,
                "separation": separation}


def main() -> None:
    client = get_client()
    cache = AnswerCache()
    seeder = Seeder(client, cache)
    curated_map = json.loads(CURATED.read_text())["votes_per_type"]

    if "--validate" in sys.argv:
        print("blind validation (name + screen state) vs curated votes:")
        groups = {"with-screen": [0, 0], "name-only": [0, 0]}
        for name, vote in curated_map.items():
            r = seeder.judge(name)
            hit = r["behavior"] == vote["behavior"]
            g = "with-screen" if name in SCREEN_BY_TYPE else "name-only"
            groups[g][0] += hit
            groups[g][1] += 1
            print(f"  {name:<8} curated={vote['behavior']:<10} "
                  f"model={r['behavior']:<10} conf={r['confidence']:.2f} "
                  f"sep={r['separation']:.2f} {'HIT' if hit else 'MISS'}"
                  + ("" if g == "with-screen" else "  (no screen data)"))
        for g, (ok, n) in groups.items():
            print(f"agreement[{g}]: {ok}/{n}")

    types = json.loads(ARTIFACT.read_text())
    shaped = [t for t in types if DN_SHAPED.match(t)]
    unshaped = [t for t in types if not DN_SHAPED.match(t)]
    todo = [t for t in shaped if t not in curated_map]
    print(f"{len(types)} types: {len(shaped)} DN-shaped, {len(unshaped)} "
          f"unshaped (skipped), {len(todo)} to seed")

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(seeder.judge, todo))

    seeds, review = {}, {}
    for name, r in zip(todo, results):
        if (r["family"] == "no-clear-role"
                or name not in SCREEN_BY_TYPE
                or r["confidence"] < SEED_CONFIDENT):
            review[name] = {"family_hint": r["family"],
                            "confidence": r["confidence"],
                            "separation": r["separation"],
                            "has_screen": name in SCREEN_BY_TYPE}
        else:
            seeds[name] = {"behavior": r["behavior"], "w": None,
                           "confidence": r["confidence"],
                           "evidence": f"jev-seed (family={r['family']}, "
                                       f"sep={r['separation']:.2f})",
                           "confidence_label": "model"}
    cache.save()
    OUT.write_text(json.dumps({
        "seeds": seeds, "review": review,
        "skipped_unshaped": unshaped,
        "meta": {"source": "jev-latest two-level DN hierarchy",
                 "confidence_threshold": SEED_CONFIDENT,
                 "calibration": "curated-map validation: screen-evidenced "
                                "votes reproduced 5/5 at conf>=0.9; sub-0.9 "
                                "band 1/5 (misses all <=0.71); name-only "
                                "channel review-only (2/5)",
                 "note": "curation seed only; weights are null until a "
                         "curator sets them; not loaded by the sim"}},
        indent=1))
    n_no_role = sum(1 for r in review.values() if r["family_hint"] == "no-clear-role")
    print(f"seeds: {len(seeds)}, review: {len(review)} "
          f"({n_no_role} of them no-clear-role), wrote {OUT}")


if __name__ == "__main__":
    main()
