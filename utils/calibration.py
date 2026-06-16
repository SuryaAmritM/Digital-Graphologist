from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

from utils.analysis import analyze_hanging_attachment, analyze_shirorekha
from utils.preprocessing import adaptive_threshold_ink
from utils.scoring import classify_risk, compute_structural_integrity_score


@dataclass
class CalibrationRow:
    image_path: str
    expected_band: str


def load_targets(csv_path: str) -> list[CalibrationRow]:
    rows: list[CalibrationRow] = []
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                CalibrationRow(
                    image_path=row["image_path"].strip(),
                    expected_band=row["expected_band"].strip(),
                )
            )
    return rows


def evaluate_targets(targets: list[CalibrationRow]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for row in targets:
        path = Path(row.image_path)
        image = cv2.imread(str(path))
        if image is None:
            results.append(
                {
                    "image_path": row.image_path,
                    "expected_band": row.expected_band,
                    "predicted_band": "Unreadable",
                    "score": -1,
                    "match": False,
                }
            )
            continue

        binary = adaptive_threshold_ink(image)
        rule_a = analyze_shirorekha(binary)
        rule_b = analyze_hanging_attachment(binary, rule_a)
        report = compute_structural_integrity_score(rule_a, rule_b)
        predicted_band = classify_risk(report["score"])["label"]

        results.append(
            {
                "image_path": row.image_path,
                "expected_band": row.expected_band,
                "predicted_band": predicted_band,
                "score": report["score"],
                "match": predicted_band == row.expected_band,
            }
        )
    return results


def print_report(results: list[dict[str, Any]]) -> None:
    total = len(results)
    correct = sum(1 for row in results if row["match"])
    accuracy = (correct / total * 100.0) if total else 0.0

    print(f"Calibration samples: {total}")
    print(f"Band match accuracy: {correct}/{total} ({accuracy:.1f}%)")
    print("-" * 72)
    for row in results:
        print(
            f"{row['image_path']}: expected={row['expected_band']}, "
            f"predicted={row['predicted_band']}, score={row['score']}"
        )


if __name__ == "__main__":
    targets = load_targets("samples/calibration_targets.csv")
    report = evaluate_targets(targets)
    print_report(report)
