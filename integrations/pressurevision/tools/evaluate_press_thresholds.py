#!/usr/bin/env python3
"""Freeze force-level thresholds on one session and score them on another.

The separation reported so far is a fit, not a result: the thresholds would be
read from the same eight trials per level they were scored on. This turns it
into a result by fitting and scoring on disjoint data.

Three modes:

- `--fit A --test B` transfers thresholds between sessions. This is the
  strictest test and, so far, the one the data fails: absolute levels drift
  between sittings even though each session separates internally.
- `--fit A` alone runs leave-one-trial-out cross-validation inside A. Honest
  about not being a fresh session, but it costs no new recording.
- `--calibrate-trials N` fits on the first N presses of each level *within* the
  evaluated session and scores the rest. This models the operator pressing a
  few times before starting, and on the two clean sessions one press per level
  is enough.

Decisions are made per trial, on the median of the frames in a held press,
because the frames inside one press are not independent samples.

Read-only. Drives nothing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METRICS = ("contact_px", "sum_kpa", "mean_kpa_in_contact", "max_kpa")
DEFAULT_METRIC = "contact_px"
ABSTAIN = "abstain"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--test", type=Path, default=None)
    parser.add_argument(
        "--metric",
        default=DEFAULT_METRIC,
        choices=list(METRICS) + ["all"],
        help="feature to threshold (default contact_px, the one that separated)",
    )
    parser.add_argument(
        "--abstain-fraction",
        type=float,
        default=0.5,
        help=(
            "width of the abstain band as a fraction of the gap between "
            "adjacent levels in the fit data. 0 disables abstaining."
        ),
    )
    parser.add_argument(
        "--calibrate-trials",
        type=int,
        default=0,
        help=(
            "instead of transferring thresholds between sessions, fit on the "
            "first N trials of each level within the evaluated session and "
            "score the rest. Absolute levels drift between sessions while "
            "within-session separation holds, so this models an operator "
            "pressing a few times to calibrate before starting."
        ),
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    if not 0.0 <= args.abstain_fraction <= 1.0:
        parser.error("--abstain-fraction must be in [0, 1]")
    if args.calibrate_trials < 0:
        parser.error("--calibrate-trials must not be negative")
    return args


def load_trials(session: Path, metric: str) -> tuple[dict[int, np.ndarray], dict]:
    """One value per held press, keyed by target grams."""
    rows = [
        json.loads(line)
        for line in (session / "capture.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    meta = rows[0]
    trials: dict[tuple[int, int], list[float]] = {}
    for row in rows:
        if row.get("row_type") != "sample":
            continue
        trials.setdefault(
            (row["trial_index"], row["target_g"]), []
        ).append(float(row[metric]))
    per_level: dict[int, list[float]] = {g: [] for g in meta["targets_g"]}
    for (_index, grams), values in sorted(trials.items()):
        per_level[grams].append(float(np.median(values)))
    return {g: np.asarray(v) for g, v in per_level.items()}, meta


def fit_thresholds(per_level: dict[int, np.ndarray], abstain_fraction: float) -> dict:
    """Boundaries between adjacent levels, with an abstain band around each.

    Assumes the feature increases with force. Where the fit levels are cleanly
    separated the boundary sits in the middle of the gap; where they overlap it
    sits between the medians and the band covers part of the overlap, so an
    overlapping fit produces abstentions rather than confident errors.
    """
    levels = sorted(per_level)
    boundaries = []
    for low, high in zip(levels, levels[1:]):
        a, b = per_level[low], per_level[high]
        if a.size == 0 or b.size == 0:
            raise ValueError(f"no fit trials for {low} g or {high} g")
        separated = bool(a.max() < b.min())
        if separated:
            edge = (float(a.max()) + float(b.min())) / 2.0
            gap = float(b.min()) - float(a.max())
        else:
            edge = (float(np.median(a)) + float(np.median(b))) / 2.0
            gap = float(a.max()) - float(b.min())  # the overlap extent
        half_band = abstain_fraction * abs(gap) / 2.0
        boundaries.append(
            {
                "below_g": low,
                "above_g": high,
                "edge": edge,
                "gap": gap,
                "separated_in_fit": separated,
                "abstain_lo": edge - half_band,
                "abstain_hi": edge + half_band,
            }
        )
    return {"levels": levels, "boundaries": boundaries}


def classify(value: float, thresholds: dict):
    """Return the predicted target in grams, or ABSTAIN inside a band."""
    for boundary in thresholds["boundaries"]:
        # A zero-width band must never fire, or a value landing exactly on the
        # edge would abstain even with abstaining switched off.
        if (
            boundary["abstain_hi"] > boundary["abstain_lo"]
            and boundary["abstain_lo"] <= value <= boundary["abstain_hi"]
        ):
            return ABSTAIN
    level = thresholds["levels"][0]
    for boundary in thresholds["boundaries"]:
        if value > boundary["edge"]:
            level = boundary["above_g"]
    return level


def score(per_level: dict[int, np.ndarray], thresholds: dict) -> dict:
    levels = thresholds["levels"]
    confusion = {str(t): {str(p): 0 for p in levels + [ABSTAIN]} for t in levels}
    decided = correct = abstained = total = 0
    for truth, values in per_level.items():
        for value in values:
            prediction = classify(float(value), thresholds)
            confusion[str(truth)][str(prediction)] += 1
            total += 1
            if prediction == ABSTAIN:
                abstained += 1
            else:
                decided += 1
                correct += int(prediction == truth)
    return {
        "trials": total,
        "abstained": abstained,
        "abstain_rate": abstained / total if total else None,
        "decided": decided,
        "correct": correct,
        "accuracy_on_decided": correct / decided if decided else None,
        "accuracy_counting_abstain_as_wrong": correct / total if total else None,
        "confusion": confusion,
    }


def leave_one_out(per_level: dict[int, np.ndarray], abstain_fraction: float) -> dict:
    """Refit without each trial in turn, so no trial scores its own threshold."""
    levels = sorted(per_level)
    held_out = {level: [] for level in levels}
    for level in levels:
        for index in range(per_level[level].size):
            reduced = {
                other: (
                    np.delete(per_level[other], index)
                    if other == level
                    else per_level[other]
                )
                for other in levels
            }
            if reduced[level].size == 0:
                continue
            thresholds = fit_thresholds(reduced, abstain_fraction)
            held_out[level].append(
                classify(float(per_level[level][index]), thresholds)
            )
    confusion = {str(t): {str(p): 0 for p in levels + [ABSTAIN]} for t in levels}
    decided = correct = abstained = total = 0
    for level, predictions in held_out.items():
        for prediction in predictions:
            confusion[str(level)][str(prediction)] += 1
            total += 1
            if prediction == ABSTAIN:
                abstained += 1
            else:
                decided += 1
                correct += int(prediction == level)
    return {
        "trials": total,
        "abstained": abstained,
        "abstain_rate": abstained / total if total else None,
        "decided": decided,
        "correct": correct,
        "accuracy_on_decided": correct / decided if decided else None,
        "accuracy_counting_abstain_as_wrong": correct / total if total else None,
        "confusion": confusion,
    }


def calibrate_within_session(per_level, calibrate_trials: int, abstain_fraction: float):
    """Fit on the first N presses of each level, score the remainder."""
    fit = {g: v[:calibrate_trials] for g, v in per_level.items()}
    test = {g: v[calibrate_trials:] for g, v in per_level.items()}
    for grams, values in fit.items():
        if values.size < calibrate_trials:
            raise SystemExit(f"only {values.size} trials at {grams} g to calibrate on")
    thresholds = fit_thresholds(fit, abstain_fraction)
    return thresholds, score(test, thresholds)


def evaluate(args, metric: str) -> dict:
    if args.calibrate_trials > 0:
        session = args.test or args.fit
        per_level, meta = load_trials(session, metric)
        thresholds, scored = calibrate_within_session(
            per_level, args.calibrate_trials, args.abstain_fraction
        )
        return {
            "metric": metric,
            "mode": "per_session_calibration",
            "abstain_fraction": args.abstain_fraction,
            "calibrate_trials_per_level": args.calibrate_trials,
            "session": str(session),
            "targets_g": meta["targets_g"],
            "thresholds": thresholds,
            "score": scored,
        }

    fit_levels, fit_meta = load_trials(args.fit, metric)
    thresholds = fit_thresholds(fit_levels, args.abstain_fraction)
    result = {
        "metric": metric,
        "abstain_fraction": args.abstain_fraction,
        "fit_session": str(args.fit),
        "fit_targets_g": fit_meta["targets_g"],
        "fit_trials_per_level": {
            str(g): int(v.size) for g, v in fit_levels.items()
        },
        "thresholds": thresholds,
    }
    if args.test is None:
        result["mode"] = "leave_one_trial_out_within_fit_session"
        result["caveat"] = (
            "not a held-out session: same operator, surface, viewpoint and "
            "sitting, so this bounds threshold overfitting only"
        )
        result["score"] = leave_one_out(fit_levels, args.abstain_fraction)
        return result

    test_levels, test_meta = load_trials(args.test, metric)
    if sorted(test_meta["targets_g"]) != sorted(fit_meta["targets_g"]):
        raise SystemExit(
            f"target mismatch: fit {fit_meta['targets_g']} vs "
            f"test {test_meta['targets_g']}"
        )
    result["mode"] = "held_out_session"
    result["test_session"] = str(args.test)
    result["test_trials_per_level"] = {
        str(g): int(v.size) for g, v in test_levels.items()
    }
    result["score"] = score(test_levels, thresholds)
    return result


def _print(result: dict) -> None:
    score_block = result["score"]
    print(f"\n=== {result['metric']}  [{result['mode']}] ===")
    for boundary in result["thresholds"]["boundaries"]:
        flag = "" if boundary["separated_in_fit"] else "   (OVERLAPPING IN FIT)"
        print(
            f"  {boundary['below_g']:>4} | {boundary['above_g']:<4} g   "
            f"edge {boundary['edge']:9.1f}   "
            f"abstain [{boundary['abstain_lo']:.1f}, {boundary['abstain_hi']:.1f}]"
            f"{flag}"
        )
    accuracy = score_block["accuracy_on_decided"]
    print(
        f"  trials {score_block['trials']}   "
        f"abstained {score_block['abstained']} "
        f"({score_block['abstain_rate']:.1%})   "
        f"accuracy on decided "
        f"{'n/a' if accuracy is None else f'{accuracy:.1%}'}"
    )
    header = "        " + "".join(f"{p:>10}" for p in score_block["confusion"][
        next(iter(score_block["confusion"]))
    ])
    print("  confusion (rows = truth):")
    print("  " + header)
    for truth, predictions in score_block["confusion"].items():
        line = f"  {truth:>6} " + "".join(f"{n:>10}" for n in predictions.values())
        print(line)


def main(argv=None) -> int:
    args = parse_args(argv)
    metrics = list(METRICS) if args.metric == "all" else [args.metric]
    results = [evaluate(args, metric) for metric in metrics]
    for result in results:
        _print(result)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
