"""Streamlit web demo for Mahalanobis Concept Drift Detector."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st

from src.mcd.io import load_labeled_tickets_csv, resolve_dataset_path
from src.mcd.modeling.classifier import MahalanobisDriftDetector

st.set_page_config(page_title="MCD Drift Detector", layout="wide")

DEFAULT_DISK_MODEL = "models/demo_stand/demo_model.joblib"
ARCHIVE_ZIP = "data/archive.zip"
DEMO_CSV = "data/demo_labeled.csv"


def main() -> None:
    st.title("Mahalanobis Concept Drift Detector")

    with st.sidebar:
        st.header("Model Management")
        mode = st.radio(
            "Choose mode:",
            [
                "Load from disk",
                "Upload Model (.joblib only)",
                "Train Model",
            ],
            help="Модель с проектором загружайте с диска: рядом с .joblib должны быть *_projector.pt и *_projector_config.json.",
        )

        detector: MahalanobisDriftDetector | None = None

        if mode == "Load from disk":
            path_in = st.text_input(
                "Path to .joblib",
                value=DEFAULT_DISK_MODEL,
                help="Относительно текущей рабочей директории (запускайте Streamlit из корня репозитория).",
            )
            if st.button("Load model"):
                p = Path(path_in)
                if not p.is_file():
                    st.error(f"File not found: {p.resolve()}")
                else:
                    try:
                        detector = MahalanobisDriftDetector.load(str(p))
                        st.success("Model loaded (with projector if artifacts present).")
                    except Exception as e:
                        st.error(f"Failed to load: {e}")

        elif mode == "Upload Model (.joblib only)":
            uploaded_file = st.file_uploader("Upload trained model (.joblib)", type=["joblib"])
            if uploaded_file:
                with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    tmp_path = tmp.name
                try:
                    detector = MahalanobisDriftDetector.load(tmp_path)
                    st.success("Model loaded.")
                except Exception as e:
                    st.warning(
                        "If this model used a projector, upload is not supported — "
                        f"use «Load from disk» instead. ({e})"
                    )
                finally:
                    os.unlink(tmp_path)

        else:  # Train Model
            st.subheader("Training Parameters")
            label_col = st.selectbox("Label Column", ["queue", "tag_1"])
            threshold_q = st.slider("Threshold Quantile", 0.90, 1.0, 0.99, 0.01)
            min_size = st.number_input("Min Cluster Size", 1, 50, 10)
            seed_val = st.number_input("Seed", 0, 1000, 42)

            if st.button("Train Model"):
                try:
                    with st.spinner("Loading data..."):
                        if Path(ARCHIVE_ZIP).is_file():
                            csv_path, _ = resolve_dataset_path(ARCHIVE_ZIP)
                        elif Path(DEMO_CSV).is_file():
                            csv_path = DEMO_CSV
                        else:
                            raise FileNotFoundError(
                                f"Neither {ARCHIVE_ZIP} nor {DEMO_CSV} found. "
                                "Add data or run scripts/run_demo_stand_with_projector.py."
                            )
                    with st.spinner("Training..."):
                        texts, labels, _, _ = load_labeled_tickets_csv(csv_path, label_col)
                        det = MahalanobisDriftDetector(
                            threshold_quantile=threshold_q,
                            min_cluster_size=min_size,
                        )
                        np.random.seed(int(seed_val))
                        max_samples = min(5000, len(texts))
                        idx = np.random.choice(len(texts), max_samples, replace=False)
                        train_texts = [texts[i] for i in idx]
                        train_labels = [labels[i] for i in idx]
                        det.fit(train_texts, train_labels)
                        os.makedirs("models", exist_ok=True)
                        det.save("models/demo_model.joblib")
                    detector = det
                    st.success("Model trained and saved to models/demo_model.joblib (no projector).")
                except Exception as e:
                    st.error(f"Training failed: {e}")

        if detector is not None:
            st.session_state.detector = detector

    st.header("Predict Drift")

    detector = st.session_state.get("detector")

    if detector is None:
        st.warning(
            "No model loaded. Use «Load from disk» after "
            "`python scripts/run_demo_stand_with_projector.py`, or train/upload in the sidebar."
        )
        return

    input_text = st.text_area("Enter customer support ticket text:")

    if st.button("Predict"):
        if not input_text.strip():
            st.error("Please enter some text.")
        else:
            try:
                pred_label, distance, threshold, is_drift = detector.predict(input_text)
                score = distance - threshold

                col1, col2, col3 = st.columns(3)
                col1.metric("Predicted Label", pred_label)
                col2.metric("Distance", f"{distance:.4f}")
                col3.metric("Threshold", f"{threshold:.4f}")

                if is_drift:
                    st.error(f"**DRIFT DETECTED** (score: {score:.4f})")
                else:
                    st.success(f"No drift (score: {score:.4f})")
            except Exception as e:
                st.error(f"Prediction failed: {e}")


if __name__ == "__main__":
    main()
