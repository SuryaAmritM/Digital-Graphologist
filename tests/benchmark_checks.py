from __future__ import annotations

from time import perf_counter

import cv2

from utils.analysis import analyze_hanging_attachment, analyze_shirorekha
from utils.preprocessing import adaptive_threshold_ink
from utils.scoring import compute_structural_integrity_score
from utils.skeleton import extract_wireframe


def run_pipeline(image_path: str) -> tuple[int, float]:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    start = perf_counter()
    binary = adaptive_threshold_ink(image)
    _ = extract_wireframe(binary)
    rule_a = analyze_shirorekha(binary)
    rule_b = analyze_hanging_attachment(binary, rule_a)
    score = compute_structural_integrity_score(rule_a, rule_b)["score"]
    elapsed = perf_counter() - start
    return score, elapsed


def main() -> None:
    cases = {
        "normal": "data/raw/normal_clean.jpeg",
        "noisy": "data/raw/noisy_shadows.jpeg",
        "tremor": "data/raw/tremor_broken.jpeg",
    }

    results: dict[str, tuple[int, float]] = {}
    for name, path in cases.items():
        score, elapsed = run_pipeline(path)
        results[name] = (score, elapsed)
        print(f"{name}: score={score}, time={elapsed:.3f}s")

    normal_score = results["normal"][0]
    noisy_score = results["noisy"][0]
    tremor_score = results["tremor"][0]

    if not (normal_score >= noisy_score >= tremor_score):
        raise AssertionError(
            "Expected ordering normal >= noisy >= tremor was not satisfied."
        )

    slow_cases = [name for name, (_, t) in results.items() if t >= 5.0]
    if slow_cases:
        raise AssertionError(f"Runtime exceeded 5s for: {', '.join(slow_cases)}")

    print("Benchmark checks passed.")


if __name__ == "__main__":
    main()
