"""Interactive, evidence-preserving review for one PV demonstration episode."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import cv2


OUTCOME_NO_SLIP = "success_no_slip"
OUTCOME_RECOVERED = "success_recovered_slip"
OUTCOME_FAILURE = "failure"
TRAINING_OUTCOMES = frozenset((OUTCOME_NO_SLIP, OUTCOME_RECOVERED))
FAILURE_REASONS = frozenset(("full_detach", "regrasp", "task_incomplete"))
TABLE_CONTACTS = ("none", "brief", "supported")
REVIEW_FPS = 10.0
UNSET_LABEL_COLOR = (170, 170, 170)
GOOD_LABEL_COLOR = (0, 220, 0)
RECOVERED_LABEL_COLOR = (255, 255, 0)
WARNING_LABEL_COLOR = (0, 165, 255)
FAILURE_LABEL_COLOR = (0, 0, 255)
SLIP_LABEL_COLOR = (255, 0, 255)
HELP_COLOR = (0, 255, 255)


@dataclass(frozen=True)
class ReviewFrame:
    timestamp_s: float
    front_path: Path
    side_path: Path
    commanded_gripper_pos: float
    observed_gripper_pos: float
    pv_teacher: float
    pv_valid: bool


def teacher_tighten_candidate(frames: list[ReviewFrame]) -> int | None:
    """Return the first post-q32 tightening frame; this is not a slip label."""
    reached_zero = False
    for index, frame in enumerate(frames):
        if abs(frame.commanded_gripper_pos - 32.0) <= 0.5:
            reached_zero = True
        elif reached_zero and frame.commanded_gripper_pos < 31.5:
            return index
    return None


def validate_decision(
    *,
    outcome: str | None,
    slip_index: int | None,
    stable_index: int | None,
    failure_reasons: set[str],
    table_contact: str | None,
    residual_grade: int | None,
    functional_damage: bool | None,
    frame_count: int,
) -> str | None:
    if outcome not in {OUTCOME_NO_SLIP, OUTCOME_RECOVERED, OUTCOME_FAILURE}:
        return "select S, R, or F"
    if residual_grade not in {0, 1, 2}:
        return "select residual grade 0, 1, or 2"
    if functional_damage is None:
        return "select functional damage Y or N"
    if table_contact not in TABLE_CONTACTS:
        return "select table contact with T: none, brief, or supported"
    if not failure_reasons <= FAILURE_REASONS:
        return "unknown failure reason"
    for index in (slip_index, stable_index):
        if index is not None and not 0 <= index < frame_count:
            return "event marker is outside the episode"
    if outcome == OUTCOME_NO_SLIP and (slip_index is not None or stable_index is not None):
        return "S cannot retain slip/stable markers; press C to clear them"
    if outcome != OUTCOME_FAILURE and failure_reasons:
        return "S/R cannot include a terminal failure flag"
    if outcome != OUTCOME_FAILURE and residual_grade == 2:
        return "residual grade 2 must be labelled F"
    if outcome != OUTCOME_FAILURE and functional_damage:
        return "functional damage must be labelled F"
    if outcome == OUTCOME_RECOVERED:
        if slip_index is None or stable_index is None:
            return "R requires J=first slip and K=first stable"
        if stable_index <= slip_index:
            return "first stable must follow first slip"
    if (
        outcome == OUTCOME_FAILURE
        and not failure_reasons
        and residual_grade != 2
        and not functional_damage
    ):
        return "F requires D/G/U, residual grade 2, or functional damage"
    if slip_index is not None and stable_index is not None and stable_index <= slip_index:
        return "first stable must follow first slip"
    return None


def outcome_record(
    *,
    attempt: int,
    dataset_episode: int | None,
    outcome: str,
    slip_index: int | None,
    stable_index: int | None,
    failure_reasons: set[str],
    table_contact: str,
    residual_grade: int,
    functional_damage: bool,
    frames: list[ReviewFrame],
    review_video: Path,
    review_timeline: Path,
    evidence_root: Path,
) -> dict:
    error = validate_decision(
        outcome=outcome,
        slip_index=slip_index,
        stable_index=stable_index,
        failure_reasons=failure_reasons,
        table_contact=table_contact,
        residual_grade=residual_grade,
        functional_damage=functional_damage,
        frame_count=len(frames),
    )
    if error is not None:
        raise ValueError(error)
    candidate = teacher_tighten_candidate(frames)
    record = {
        "schema_version": 1,
        "attempt": int(attempt),
        "dataset_episode": dataset_episode,
        "outcome": outcome,
        "promoted_to_training": outcome in TRAINING_OUTCOMES,
        "failure_reasons": sorted(failure_reasons),
        "table_contact": table_contact,
        "residual_grade": int(residual_grade),
        "functional_damage": bool(functional_damage),
        "first_slip_s": None if slip_index is None else frames[slip_index].timestamp_s,
        "first_stable_s": None if stable_index is None else frames[stable_index].timestamp_s,
        "recovery_time_s": (
            None
            if slip_index is None or stable_index is None
            else frames[stable_index].timestamp_s - frames[slip_index].timestamp_s
        ),
        "teacher_tighten_candidate_s": (
            None if candidate is None else frames[candidate].timestamp_s
        ),
        "q_stable_command": None,
        "q_min_hold_command": None,
        "extra_closure_command": None,
        "review_video": str(review_video.relative_to(evidence_root)),
        "review_timeline": str(review_timeline.relative_to(evidence_root)),
        "annotation_method": "operator_confirmed_review_video",
    }
    if stable_index is not None:
        q_stable = frames[stable_index].commanded_gripper_pos
        q_min_hold = min(frame.commanded_gripper_pos for frame in frames[stable_index:])
        record.update(
            {
                "q_stable_command": q_stable,
                "q_min_hold_command": q_min_hold,
                "extra_closure_command": q_stable - q_min_hold,
            }
        )
    return record


def _read_pair(frame: ReviewFrame):
    front = cv2.imread(str(frame.front_path), cv2.IMREAD_COLOR)
    side = cv2.imread(str(frame.side_path), cv2.IMREAD_COLOR)
    if front is None:
        raise RuntimeError(f"could not read review frame: {frame.front_path}")
    if side is None:
        raise RuntimeError(f"could not read review frame: {frame.side_path}")
    if front.shape[0] != side.shape[0]:
        raise RuntimeError("front and side review frames must have equal height")
    return front, side


def review_status_tokens(
    *,
    outcome: str | None,
    residual_grade: int | None,
    functional_damage: bool | None,
    table_contact: str | None,
    slip_index: int | None,
    stable_index: int | None,
    failure_reasons: set[str],
) -> tuple[tuple[str, tuple[int, int, int]], ...]:
    outcome_colors = {
        OUTCOME_NO_SLIP: GOOD_LABEL_COLOR,
        OUTCOME_RECOVERED: RECOVERED_LABEL_COLOR,
        OUTCOME_FAILURE: FAILURE_LABEL_COLOR,
    }
    grade_colors = {
        0: GOOD_LABEL_COLOR,
        1: WARNING_LABEL_COLOR,
        2: FAILURE_LABEL_COLOR,
    }
    table_colors = {
        "none": GOOD_LABEL_COLOR,
        "brief": WARNING_LABEL_COLOR,
        "supported": FAILURE_LABEL_COLOR,
    }
    damage_color = (
        UNSET_LABEL_COLOR
        if functional_damage is None
        else FAILURE_LABEL_COLOR if functional_damage else GOOD_LABEL_COLOR
    )
    failures = ",".join(sorted(failure_reasons)) or "-"
    return (
        (f"outcome={outcome or '-'}", outcome_colors.get(outcome, UNSET_LABEL_COLOR)),
        (
            f"grade={residual_grade if residual_grade is not None else '-'}",
            grade_colors.get(residual_grade, UNSET_LABEL_COLOR),
        ),
        (
            f"damage={'Y' if functional_damage else 'N' if functional_damage is False else '-'}",
            damage_color,
        ),
        (
            f"table={table_contact or '-'}",
            table_colors.get(table_contact, UNSET_LABEL_COLOR),
        ),
        (
            f"slip={slip_index if slip_index is not None else '-'}",
            SLIP_LABEL_COLOR if slip_index is not None else UNSET_LABEL_COLOR,
        ),
        (
            f"stable={stable_index if stable_index is not None else '-'}",
            RECOVERED_LABEL_COLOR if stable_index is not None else UNSET_LABEL_COLOR,
        ),
        (
            f"failures={failures}",
            FAILURE_LABEL_COLOR if failure_reasons else UNSET_LABEL_COLOR,
        ),
    )


def _draw_colored_tokens(
    image,
    tokens: tuple[tuple[str, tuple[int, int, int]], ...],
    *,
    x: int,
    y: int,
) -> int:
    line_height = 26
    cursor_x = x
    cursor_y = y
    for text, color in tokens:
        (width, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        if cursor_x > x and cursor_x + width > image.shape[1] - x:
            cursor_x = x
            cursor_y += line_height
        cv2.putText(
            image,
            text,
            (cursor_x, cursor_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
        cursor_x += width + 20
    return cursor_y


def write_review_artifacts(
    frames: list[ReviewFrame], evidence_root: Path, *, attempt: int
) -> tuple[Path, Path]:
    if not frames:
        raise ValueError("cannot review an empty episode")
    review_dir = evidence_root / "episode_reviews"
    review_dir.mkdir(exist_ok=True)
    stem = f"attempt-{attempt:06d}"
    video_path = review_dir / f"{stem}.avi"
    timeline_path = review_dir / f"{stem}.csv"
    front, side = _read_pair(frames[0])
    frame_size = (front.shape[1] + side.shape[1], front.shape[0])
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), REVIEW_FPS, frame_size
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open review video: {video_path}")
    candidate = teacher_tighten_candidate(frames)
    try:
        with timeline_path.open("x", newline="", encoding="utf-8") as handle:
            csv_writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "frame_index",
                    "timestamp_s",
                    "commanded_gripper_pos",
                    "observed_gripper_pos",
                    "pv_teacher",
                    "pv_valid",
                    "teacher_tighten_candidate",
                ),
                lineterminator="\n",
            )
            csv_writer.writeheader()
            for index, item in enumerate(frames):
                front, side = _read_pair(item)
                combined = cv2.hconcat((front, side))
                cv2.putText(
                    combined,
                    (
                        f"t={item.timestamp_s:.1f}s  q={item.commanded_gripper_pos:.2f}  "
                        f"read={item.observed_gripper_pos:.2f}  PV={item.pv_teacher:.2f}"
                    ),
                    (12, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                writer.write(combined)
                csv_writer.writerow(
                    {
                        "frame_index": index,
                        "timestamp_s": f"{item.timestamp_s:.6f}",
                        "commanded_gripper_pos": item.commanded_gripper_pos,
                        "observed_gripper_pos": item.observed_gripper_pos,
                        "pv_teacher": item.pv_teacher,
                        "pv_valid": int(item.pv_valid),
                        "teacher_tighten_candidate": int(index == candidate),
                    }
                )
    finally:
        writer.release()
    return video_path, timeline_path


def interactive_review(
    video_path: Path,
    frames: list[ReviewFrame],
    *,
    key_source=None,
    show: bool = True,
) -> dict | None:
    """Review with single-key decisions; Escape returns None without promotion."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open review video: {video_path}")
    window = "PV episode review"
    if show:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, 1600, 650)
    index = 0
    playing = True
    slip_index = None
    stable_index = None
    outcome = None
    failure_reasons: set[str] = set()
    table_contact = None
    residual_grade = None
    functional_damage = None
    message = ""
    candidate = teacher_tighten_candidate(frames)
    try:
        while True:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, image = capture.read()
            if not ok:
                raise RuntimeError(f"could not read review frame {index}: {video_path}")
            help_lines = (
                "SPACE play/pause/replay  ,/. seek  J slip  K stable  C clear markers",
                "S no-slip success  R recovered success  F failure  D/G/U detached/regrasp/incomplete",
                "T table contact none/brief/supported  0/1/2 residual  Y/N damage  ENTER accept",
            )
            for line_index, line in enumerate(help_lines):
                cv2.putText(
                    image,
                    line,
                    (12, 55 + 26 * line_index),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    HELP_COLOR,
                    2,
                    cv2.LINE_AA,
                )
            status_y = _draw_colored_tokens(
                image,
                review_status_tokens(
                    outcome=outcome,
                    residual_grade=residual_grade,
                    functional_damage=functional_damage,
                    table_contact=table_contact,
                    slip_index=slip_index,
                    stable_index=stable_index,
                    failure_reasons=failure_reasons,
                ),
                x=12,
                y=55 + 26 * len(help_lines),
            )
            if candidate is not None:
                status_y += 26
                cv2.putText(
                    image,
                    f"auto candidate: teacher starts tightening near frame {candidate} (not a slip label)",
                    (12, status_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    WARNING_LABEL_COLOR,
                    2,
                    cv2.LINE_AA,
                )
            if message:
                status_y += 26
                cv2.putText(
                    image,
                    message,
                    (12, status_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    FAILURE_LABEL_COLOR,
                    2,
                    cv2.LINE_AA,
                )
            marker = None
            if index == slip_index:
                marker = ("J  FIRST SLIP", SLIP_LABEL_COLOR)
            if index == stable_index:
                marker = ("K  FIRST STABLE", RECOVERED_LABEL_COLOR)
            if marker is not None:
                cv2.putText(
                    image,
                    marker[0],
                    (12, image.shape[0] - 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.85,
                    marker[1],
                    3,
                    cv2.LINE_AA,
                )
            if show:
                cv2.imshow(window, image)
            key = (
                key_source()
                if key_source is not None
                else cv2.waitKey(100 if playing else 30) & 0xFF
            )
            message = ""
            if key == 27:
                return None
            if key in (10, 13):
                message = validate_decision(
                    outcome=outcome,
                    slip_index=slip_index,
                    stable_index=stable_index,
                    failure_reasons=failure_reasons,
                    table_contact=table_contact,
                    residual_grade=residual_grade,
                    functional_damage=functional_damage,
                    frame_count=len(frames),
                ) or ""
                if not message:
                    return {
                        "outcome": outcome,
                        "slip_index": slip_index,
                        "stable_index": stable_index,
                        "failure_reasons": failure_reasons,
                        "table_contact": table_contact,
                        "residual_grade": residual_grade,
                        "functional_damage": functional_damage,
                    }
            elif key == ord(" "):
                if not playing and index == len(frames) - 1:
                    # The normal playback stops on the final frame.  Restart at
                    # frame zero instead of immediately stopping there again.
                    index = -1
                    playing = True
                else:
                    playing = not playing
            elif key == ord(","):
                playing = False
                index = max(0, index - 5)
            elif key == ord("."):
                playing = False
                index = min(len(frames) - 1, index + 5)
            elif key in (ord("j"), ord("J")):
                slip_index = index
            elif key in (ord("k"), ord("K")):
                stable_index = index
            elif key in (ord("c"), ord("C")):
                slip_index = stable_index = None
            elif key in (ord("s"), ord("S")):
                outcome = OUTCOME_NO_SLIP
            elif key in (ord("r"), ord("R")):
                outcome = OUTCOME_RECOVERED
            elif key in (ord("f"), ord("F")):
                outcome = OUTCOME_FAILURE
            elif key in (ord("d"), ord("D")):
                failure_reasons.symmetric_difference_update(("full_detach",))
            elif key in (ord("g"), ord("G")):
                failure_reasons.symmetric_difference_update(("regrasp",))
            elif key in (ord("t"), ord("T")):
                if table_contact is None:
                    table_contact = TABLE_CONTACTS[0]
                else:
                    table_contact = TABLE_CONTACTS[
                        (TABLE_CONTACTS.index(table_contact) + 1) % len(TABLE_CONTACTS)
                    ]
            elif key in (ord("u"), ord("U")):
                failure_reasons.symmetric_difference_update(("task_incomplete",))
            elif key in (ord("0"), ord("1"), ord("2")):
                residual_grade = int(chr(key))
            elif key in (ord("y"), ord("Y")):
                functional_damage = True
            elif key in (ord("n"), ord("N")):
                functional_damage = False
            elif key < 0 and key_source is not None:
                raise RuntimeError("review key source ended before a decision was accepted")
            if playing:
                index += 1
                if index >= len(frames):
                    index = len(frames) - 1
                    playing = False
    finally:
        capture.release()
        if show:
            cv2.destroyWindow(window)
