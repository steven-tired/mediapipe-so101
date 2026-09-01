"""Read back what the snapshot keys captured, and say whether the levels split.

`aim_pad_camera.py` writes one row and two frames per labelled press. This
turns that into the comparison that matters: does the metric actually separate
a light press from a hard one, on this crop, with this camera?

The judgement is d' -- the gap between the two levels over their pooled spread
-- because that is what `calibration_gate.py` gates on and what §3.11 shows
predicts whether classification survives. A large-looking gap between two
noisy levels is not a signal.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics as st

METRICS = ("mean_kpa_in_contact", "sum_kpa", "contact_px", "max_kpa")


def dprime(low, high) -> float:
    if len(low) < 2 or len(high) < 2:
        return float("nan")
    pooled = ((st.pvariance(low) + st.pvariance(high)) / 2) ** 0.5
    gap = st.median(high) - st.median(low)
    if pooled == 0:
        return float("inf") if gap else 0.0
    return gap / pooled


def summarise(rows) -> dict:
    by_label = {}
    for row in rows:
        by_label.setdefault(row.get("label"), []).append(row)
    report = {"counts": {k: len(v) for k, v in by_label.items()}, "metrics": {}}

    # A press that clips carries no cue, so it would poison the comparison.
    blown = [r for r in rows if r.get("blown_percent", 0) > 1.0]
    report["blown_frames"] = len(blown)

    # "Only a few spots work" is a claim about position, so report per position
    # before pooling -- pooling a spot that responds with one that does not
    # produces a middling d' that describes neither.
    by_position = {}
    for row in rows:
        entry = by_position.setdefault(row.get("position", 1), {})
        entry.setdefault(row.get("label"), []).append(
            row.get("mean_kpa_in_contact", 0.0)
        )
    report["by_position"] = {
        pos: {
            "median": {lab: st.median(v) for lab, v in labels.items()},
            "n": {lab: len(v) for lab, v in labels.items()},
            "dprime_light_hard": dprime(labels.get("light", []),
                                        labels.get("hard", [])),
        }
        for pos, labels in sorted(by_position.items())
    }

    for metric in METRICS:
        entry = {}
        for label, group in by_label.items():
            values = [r[metric] for r in group if metric in r]
            if values:
                entry[label] = {
                    "median": st.median(values),
                    "min": min(values),
                    "max": max(values),
                    "n": len(values),
                }
        light = [r[metric] for r in by_label.get("light", []) if metric in r]
        hard = [r[metric] for r in by_label.get("hard", []) if metric in r]
        none = [r[metric] for r in by_label.get("none", []) if metric in r]
        entry["dprime_light_hard"] = dprime(light, hard)
        entry["dprime_none_light"] = dprime(none, light)
        report["metrics"][metric] = entry
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", type=Path, default=Path("/tmp/pad_snapshots"))
    args = parser.parse_args(argv)

    path = args.snapshots / "snapshots.jsonl"
    if not path.exists():
        raise SystemExit(f"no snapshots at {path}")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    report = summarise(rows)
    print(json.dumps(report, indent=2, sort_keys=True))

    if report["blown_frames"]:
        print(f"\n{report['blown_frames']} snapshot(s) clipped -- those carry no "
              f"cue and should be recaptured before reading anything into this")
    d = report["metrics"]["mean_kpa_in_contact"]["dprime_light_hard"]
    print(f"\nlight vs hard on mean_kpa_in_contact: d' = {d:.2f}")
    print("separates" if d >= 2.5 else
          "does NOT separate -- see calibration_gate.MIN_DPRIME")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
