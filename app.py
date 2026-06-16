import cv2
import numpy as np
import streamlit as st
from time import perf_counter

from utils.analysis import analyze_hanging_attachment, analyze_shirorekha
from utils.preprocessing import adaptive_threshold_ink
from utils.scoring import classify_risk, compute_structural_integrity_score
from utils.skeleton import extract_wireframe


st.set_page_config(
    page_title="Digital Graphologist",
    page_icon="✍️",
    layout="wide",
)

st.title("Digital Graphologist")
st.caption("Early dysgraphia screening through geometric handwriting analysis.")


def _draw_boxes(image: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    overlay = image.copy()
    for x, y, w, h in boxes:
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return overlay

left_col, right_col = st.columns(2)

with left_col:
    st.subheader("Upload")
    uploaded_file = st.file_uploader(
        "Upload a notebook image",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG",
    )

    decoded_image = None
    upload_error = None

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()

        # Guardrail for huge files that can slow classroom usage.
        max_file_size_bytes = 10 * 1024 * 1024
        if len(file_bytes) > max_file_size_bytes:
            upload_error = "File is too large. Please upload an image under 10 MB."
        else:
            np_buffer = np.frombuffer(file_bytes, dtype=np.uint8)
            decoded_image = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
            if decoded_image is None:
                upload_error = (
                    "Could not read this file as a valid image. Please upload JPG or PNG."
                )

    if upload_error:
        st.error(upload_error)
    elif decoded_image is not None:
        st.success("Image uploaded successfully.")
        st.image(
            cv2.cvtColor(decoded_image, cv2.COLOR_BGR2RGB),
            caption="Input image",
            use_container_width=True,
        )
    else:
        st.info("Upload an image to start analysis.")

with right_col:
    st.subheader("Pipeline Status")
    preprocessing_ready = decoded_image is not None and not upload_error
    skeleton_ready = preprocessing_ready
    analysis_ready = preprocessing_ready
    scoring_ready = preprocessing_ready
    st.write(
        f"- Preprocessing: {'ready' if preprocessing_ready else 'waiting for input'}"
    )
    st.write(f"- Skeletonization: {'ready' if skeleton_ready else 'waiting for input'}")
    st.write(
        f"- Structural analysis: {'ready' if analysis_ready else 'waiting for input'}"
    )
    st.write(f"- Scoring: {'ready' if scoring_ready else 'waiting for input'}")

st.divider()
st.subheader("Result Preview")
if decoded_image is not None and not upload_error:
    t0 = perf_counter()
    binary_output = adaptive_threshold_ink(decoded_image)
    skeleton_output = extract_wireframe(binary_output)
    rule_a = analyze_shirorekha(binary_output)
    rule_b = analyze_hanging_attachment(binary_output, rule_a)
    score_report = compute_structural_integrity_score(rule_a, rule_b)
    score = score_report["score"]
    risk = classify_risk(score)

    diagnostic_boxes = [*rule_a["break_boxes"], *rule_b["floating_boxes"]]
    original_with_boxes = _draw_boxes(decoded_image, diagnostic_boxes)
    skeleton_bgr = cv2.cvtColor(skeleton_output, cv2.COLOR_GRAY2BGR)
    skeleton_with_boxes = _draw_boxes(skeleton_bgr, diagnostic_boxes)

    preview_left, preview_right = st.columns(2)
    with preview_left:
        st.image(
            cv2.cvtColor(original_with_boxes, cv2.COLOR_BGR2RGB),
            caption="Original with flagged regions",
            use_container_width=True,
        )
    with preview_right:
        st.image(
            cv2.cvtColor(skeleton_with_boxes, cv2.COLOR_BGR2RGB),
            caption="X-ray wireframe with flagged regions",
            use_container_width=True,
        )

    st.divider()
    st.subheader("Structural Analysis")
    metric_col_1, metric_col_2, metric_col_3 = st.columns(3)
    metric_col_1.metric("Shirorekha continuity", f"{rule_a['continuity_ratio'] * 100:.1f}%")
    metric_col_2.metric("Shirorekha breaks", str(rule_a["break_count"]))
    metric_col_3.metric("Floating letters", str(rule_b["floating_count"]))

    st.subheader("Structural Integrity Score")
    st.metric("Score (0-100)", str(score))
    with st.expander("How this score is built (not only Shirorekha continuity)"):
        p = score_report["penalties"]
        st.markdown(
            "The score starts at **100** and subtracts penalties from several signals:\n"
            "- **Continuity** (gaps along the headline)\n"
            "- **Breaks** (larger headline gaps)\n"
            "- **Headline instability** (skeleton wobble in the top strip)\n"
            "- **Floating bodies** (lower components not linked to the headline zone)\n"
            "- **Motor stress** (severe combined pattern on some samples)\n"
        )
        st.json(p)
    if risk["severity"] == "success":
        st.success(f"Risk Band: {risk['label']}")
    elif risk["severity"] == "warning":
        st.warning(f"Risk Band: {risk['label']}")
    else:
        st.error(f"Risk Band: {risk['label']}")
    processing_time = perf_counter() - t0
    st.caption(f"Pipeline processing time: {processing_time:.3f}s (target: < 5.0s)")

    st.subheader("Why this result?")
    st.markdown(
        (
            f"- **Top-line continuity (Shirorekha):** {rule_a['continuity_ratio'] * 100:.1f}%\n"
            f"- **Detected top-line breaks:** {rule_a['break_count']}\n"
            f"- **Headline instability (skeleton jitter):** {rule_a['top_band_instability'] * 100:.1f}%\n"
            f"- **Detected floating letter bodies:** {rule_b['floating_count']}\n\n"
            "The **final score** is not only continuity: it also reflects headline "
            "wobble (instability) and attachment of lower strokes to the headline zone. "
            "You can have **100% continuity** while still seeing a lower score if "
            "instability or floating-body flags are present."
        )
    )
else:
    st.warning("Upload a valid image to generate preprocessing output.")
