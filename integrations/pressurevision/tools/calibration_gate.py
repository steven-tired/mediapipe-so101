"""Decide whether a set of calibration presses is good enough to use.

§7 action 1 asks for a calibration step that "refuses calibration presses that
come back too close or too variable". This is that refusal, as a pure function
so it can be validated against the sessions already captured before it is ever
put in front of an operator.

Why a separation gate rather than a press count. Fitting the boundary on the
first k presses of `pv_labelled_03` and `pv_labelled_05` and scoring the rest,
the boundary between the 100 g and 330 g levels wanders with k on both
sessions and on both the min/max and the median statistic — 9.8 to 12.0 on 03,
5.6 to 9.7 on 05. More calibration presses do not settle it, because the levels
themselves drift within a session (§3.12). What decides whether that instability
matters is how far apart the levels are: session 05 separates at d' 5.47 and
scores 100% however the boundary moves, while session 03 at d' 3.11 starts
misclassifying once it shifts. Session 04, the sitting with the bad geometry,
sits at d' 1.45 and is the one a gate has to catch.

So the gate asks the question that predicts the outcome — are these levels far
enough apart relative to their own spread — and not the one that is easy to
count.
"""

from __future__ import annotations

import statistics as st

# Session 04 (bad geometry, §3.11 gives it 1.1-1.3 usable levels) reaches d'
# 1.45 on mean_kpa_in_contact; session 03 reaches 3.11 and 05 reaches 5.47,
# and both of those calibrate usefully. The gate belongs between them, near
# enough to 04 to catch a sitting that has gone wrong and far enough below 03
# not to reject a workable one.
MIN_DPRIME = 2.5
# Below this there is no spread to judge, so d' cannot be estimated and the
# gate would pass anything.
MIN_PRESSES_PER_LEVEL = 3

# There is deliberately no separate "presses too variable" criterion. One was
# written and it rejected every session captured so far, including the one that
# classifies at 100%: peak-to-peak spread is roughly three to four times the
# standard deviation d' already divides by, so the two tests measure the same
# thing at different scales and the stricter one always wins. d' alone carries
# "too close" and "too variable" together, which is what makes it the gate.


def _dprime(low, high) -> float:
    pooled = ((st.pvariance(low) + st.pvariance(high)) / 2) ** 0.5
    separation = st.median(high) - st.median(low)
    if pooled == 0:
        return float("inf") if separation > 0 else 0.0
    return separation / pooled


def check(per_level: dict, *, min_dprime=MIN_DPRIME,
          min_presses=MIN_PRESSES_PER_LEVEL) -> dict:
    """Is this calibration usable? Returns a verdict and what to redo.

    `per_level` maps a level label to that level's per-press values, in the
    order they were pressed.
    """
    levels = sorted(per_level)
    reasons = []
    if len(levels) < 2:
        return {"accepted": False, "reasons": ["fewer than two levels"],
                "boundaries": []}

    for level in levels:
        count = len(per_level[level])
        if count < min_presses:
            reasons.append(
                f"level {level}: {count} presses, need {min_presses} — with "
                f"fewer there is no spread to judge"
            )

    boundaries = []
    for low, high in zip(levels, levels[1:]):
        a, b = list(per_level[low]), list(per_level[high])
        if len(a) < 2 or len(b) < 2:
            continue
        d = _dprime(a, b)
        gap = st.median(b) - st.median(a)
        entry = {
            "below": low, "above": high, "dprime": d,
            "gap": gap, "ordered": gap > 0,
        }
        boundaries.append(entry)

        if not entry["ordered"]:
            reasons.append(
                f"{low} to {high}: the harder press did not read higher — "
                f"press the levels in the order asked"
            )
            continue
        if d < min_dprime:
            reasons.append(
                f"{low} to {high}: d' {d:.2f} below {min_dprime} — press "
                f"further apart, or hold each level steadier"
            )

    return {"accepted": not reasons, "reasons": reasons, "boundaries": boundaries}
