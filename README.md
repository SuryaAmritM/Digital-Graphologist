# Digital Graphologist

Digital Graphologist is a Streamlit-based prototype for early dysgraphia screening in handwritten Devanagari script.
It analyzes structural geometry (not OCR content) and outputs a Structural Integrity Score with risk bands.

## Current Capabilities (Phase 1 Prototype)

- Upload notebook images (`.jpg`, `.jpeg`, `.png`) in the web app.
- Preprocess images with Sauvola adaptive thresholding.
- Generate X-ray wireframes through skeletonization.
- Run Rule A (Shirorekha continuity) and Rule B (hanging attachment) checks.
- Produce:
  - Structural Integrity Score (0-100)
  - Risk band (`Normal`, `Monitor`, `High Risk`)
  - Red-box highlighted regions on original and X-ray views
  - Plain-language explanation of the score drivers

## Project Structure

- `app.py` - Streamlit application entrypoint
- `utils/preprocessing.py` - Adaptive threshold ink isolation
- `utils/skeleton.py` - Skeleton extraction (wireframe)
- `utils/analysis.py` - Rule A and Rule B structural analysis
- `utils/scoring.py` - Score calculation and risk classification
- `utils/calibration.py` - Calibration report utility for labeled samples
- `samples/calibration_targets.csv` - Calibration target template
- `tests/benchmark_checks.py` - Runtime and score-order benchmark checks

## Setup

1. Create and activate a Python 3 environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the App

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal.

## Calibration Utility

Edit `samples/calibration_targets.csv` with your labeled dataset rows:

```csv
image_path,expected_band
path/to/image1.jpeg,Normal
path/to/image2.jpeg,Monitor
path/to/image3.jpeg,High Risk
```

Run:

```bash
python -m utils.calibration
```

This prints expected vs predicted bands and match accuracy.

## Quality Checks

Run benchmark checks:

```bash
python -m tests.benchmark_checks
```

Checks included:
- Score ordering sanity (`normal >= noisy >= tremor`)
- Per-image runtime under 5 seconds

## Known Limitations (Current Prototype)

- Heuristic rules are conservative and currently over-flag some clean writing.
- Scoring weights are not yet fully tuned to DHCD/HandPD scale.
- No OCR or linguistic analysis (intentionally out of scope for Phase 1).
- Static image workflow only (no live camera stream).

## Phase 2 Roadmap

- Calibrate Rule A/Rule B thresholds with larger labeled Devanagari notebooks.
- Improve floating-letter detection with zone-aware word segmentation.
- Add robust batch evaluation against curated validation splits.
- Expand explainability panel with per-word diagnostics.
- Introduce script abstraction for future multilingual support.
