"""Streamlit web demo for Mahalanobis Concept Drift Detector."""

import streamlit as st
import pandas as pd
import numpy as np
import os
import tempfile
from src.mcd.io import load_labeled_tickets_csv, resolve_dataset_path
from src.mcd.modeling.classifier import MahalanobisDriftDetector
from src.mcd.visualization.scatter import plot_scatter_2d
from src.mcd.visualization.projection import project_2d

st.set_page_config(page_title="MCD Drift Detector", layout="wide")

@st.cache_resource
def get_detector():
    return None

def main():
    st.title("Mahalanobis Concept Drift Detector")
    
    # Sidebar: Model management
    with st.sidebar:
        st.header("Model Management")
        mode = st.radio("Choose mode:", ["Upload Model", "Train Model"])
        
        detector = None
        
        if mode == "Upload Model":
            uploaded_file = st.file_uploader("Upload trained model (.joblib)", type=['joblib'])
            if uploaded_file:
                with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    tmp_path = tmp.name
                try:
                    detector = DriftDetectorDriftDetector.load(tmp_path)
                    st.success("Model loaded successfully!")
                except Exception as e:
                    st.error(f"Failed to load model: {e}")
                finally:
                    os.unlink(tmp_path)
        
        else:  # Train Model
            st.subheader("Training Parameters")
            label_col = st.selectbox("Label Column", ["queue", "tag_1"])
            threshold_q = st.slider("Threshold Quantile", 0.90, 0.99, 0.99, 0.01)
            min_size = st.number_input("Min Cluster Size", 5, 50, 10)
            seed_val = st.number_input("Seed", 0, 1000, 42)
            
            if st.button("Train Model"):
                try:
                    with st.spinner("Loading data..."):
                        csv_path, _ = resolve_dataset_path("data/archive.zip")
                    with st.spinner("Training..."):
                        texts, labels, _, _ = load_labeled_tickets_csv(csv_path, label_col)
                        detector = MahalanobisDriftDetector(
                            threshold_quantile=threshold_q,
                            min_cluster_size=min_size
                        )
                        np.random.seed(seed_val)
                        # Train on subset for speed
                        max_samples = min(5000, len(texts))
                        idx = np.random.choice(len(texts), max_samples, replace=False)
                        train_texts = [texts[i] for i in idx]
                        train_labels = [labels[i] for i in idx]
                        detector.fit(train_texts, train_labels)
                        
                        # Save model
                        os.makedirs("models", exist_ok=True)
                        detector.save("models/demo_model.joblib")
                    st.success("Model trained and saved!")
                except Exception as e:
                    st.error(f"Training failed: {e}")
        
        # Store detector in session state
        st.session_state.detector = detector
    
    # Main panel: Prediction
    st.header("Predict Drift")
    
    detector = st.session_state.get("detector")
    
    if detector is None:
        st.warning("No model loaded. Load or train a model in the sidebar first.")
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
                    st.error(f"🚨 **DRIFT DETECTED** (score: {score:.4f})")
                else:
                    st.success(f"✅ No drift (score: {score:.4f})")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

if __name__ == "__main__":
    main()