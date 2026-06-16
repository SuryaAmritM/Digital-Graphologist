from __future__ import annotations

from typing import Any


def compute_structural_integrity_score(
    rule_a: dict[str, Any], rule_b: dict[str, Any]
) -> dict[str, Any]:
    """Compute score out of 100 from Rule A, Rule B, and headline stability."""
    base_score = 100.0

    continuity_ratio = float(rule_a.get("continuity_ratio", 0.0))
    break_count = int(rule_a.get("break_count", 0))
    instability = float(rule_a.get("top_band_instability", 0.0))
    floating_count = int(rule_b.get("floating_count", 0))
    checked_count = int(rule_b.get("checked_count", 0))

    # Rule A: continuity (segment-aware; already blended in analysis).
    continuity_penalty = (1.0 - continuity_ratio) * 8.0

    # Breaks: cap so a few real fragments do not dominate clean writing.
    break_penalty = min(6.0, break_count * 1.0)

    # Micro-tremor / waviness along the headline (skeleton-based in analysis).
    # Ignore small baseline jitter typical for neat writing.
    instability_penalty = max(0.0, instability - 0.245) * 50.0

    if checked_count > 0:
        floating_ratio = floating_count / checked_count
    else:
        floating_ratio = 0.0

    # Rule B: unattached lower bodies.
    attachment_penalty = floating_ratio * 15.0 + min(5.0, floating_count * 0.9)

    # When the headline looks structurally solid but many CCs are still marked
    # floating, attachment errors are often false (thin headline / binarization gaps).
    if (
        checked_count > 0
        and floating_ratio > 0.55
        and continuity_ratio >= 0.88
        and instability < 0.22
        and break_count <= 3
    ):
        attachment_penalty *= 0.62

    # Perfect headline continuity + no measured breaks: trust the headline signal
    # and soften secondary penalties (skeleton jitter + CC float heuristics).
    if continuity_ratio >= 0.99 and break_count == 0:
        instability_penalty *= 0.35
        attachment_penalty *= 0.58

    # Severe cases: headline instability plus many detached lower bodies together.
    motor_stress_penalty = 0.0
    if (
        checked_count > 0
        and instability > 0.38
        and floating_ratio > 0.18
    ):
        motor_stress_penalty = 22.0

    # Tremor-style breakdown: many letter-sized CCs flagged detached while the
    # headline strip looks artificially "stable" (low jitter) in the binary image.
    if (
        checked_count >= 32
        and 0.34 <= floating_ratio <= 0.48
        and instability < 0.20
        and continuity_ratio <= 0.91
    ):
        motor_stress_penalty += 30.0

    total_penalty = (
        continuity_penalty
        + break_penalty
        + instability_penalty
        + attachment_penalty
        + motor_stress_penalty
    )
    score = max(0.0, min(100.0, base_score - total_penalty))

    return {
        "score": int(round(score)),
        "penalties": {
            "continuity_penalty": round(continuity_penalty, 2),
            "break_penalty": round(break_penalty, 2),
            "instability_penalty": round(instability_penalty, 2),
            "attachment_penalty": round(attachment_penalty, 2),
            "motor_stress_penalty": round(motor_stress_penalty, 2),
            "total_penalty": round(total_penalty, 2),
        },
    }


def classify_risk(score: int) -> dict[str, str]:
    """Map score to PRD threshold bands."""
    if score >= 90:
        return {"label": "Normal", "severity": "success", "color": "green"}
    if score >= 60:
        return {"label": "Monitor", "severity": "warning", "color": "orange"}
    return {"label": "High Risk", "severity": "error", "color": "red"}
