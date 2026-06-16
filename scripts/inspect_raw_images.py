"""Print pipeline metrics for every image in data/raw (for debugging scores)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2

from utils.analysis import analyze_hanging_attachment, analyze_shirorekha
from utils.preprocessing import adaptive_threshold_ink
from utils.scoring import classify_risk, compute_structural_integrity_score

RAW = ROOT / "data" / "raw"


def main() -> None:
    files = sorted(RAW.glob("*"))
    for path in files:
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        img = cv2.imread(str(path))
        if img is None:
            print(f"{path.name}: FAILED to read")
            continue
        b = adaptive_threshold_ink(img)
        a = analyze_shirorekha(b)
        h = analyze_hanging_attachment(b, a)
        s = compute_structural_integrity_score(a, h)
        r = classify_risk(s["score"])
        fr = h["floating_count"] / h["checked_count"] if h["checked_count"] else 0.0
        print("---", path.name, "---")
        print("  score:", s["score"], "band:", r["label"])
        print(
            "  continuity:",
            round(a["continuity_ratio"], 4),
            "seg_min:",
            round(a.get("segment_min_continuity", 0), 4),
            "instab:",
            round(a["top_band_instability"], 4),
        )
        print("  breaks:", a["break_count"], "float:", h["floating_count"], "checked:", h["checked_count"], "float_ratio:", round(fr, 3))
        print("  penalties:", s["penalties"])


if __name__ == "__main__":
    main()
