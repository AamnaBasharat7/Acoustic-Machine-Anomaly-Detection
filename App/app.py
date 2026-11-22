

import streamlit as st
import os
import io
import librosa
import numpy as np
import pandas as pd
import joblib

from feature_extraction import extract_features


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Acoustic Machine Anomaly Detection",
    page_icon="🔊",
    layout="wide"
)


# ==========================================
# 2. LOAD TRAINED MODEL FILES
# ==========================================

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

try:
    model = joblib.load(os.path.join(MODEL_DIR, "one_class_svm.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler_svm.pkl"))
    threshold = float(joblib.load(os.path.join(MODEL_DIR, "threshold_svm.pkl")))
    model_feature_names = joblib.load(
        os.path.join(MODEL_DIR, "feature_names.pkl")
    )

    if len(model_feature_names) != 34:
        raise ValueError("The model must use exactly 34 features.")

except Exception as e:
    st.error(f"Failed to load model files: {e}")
    st.stop()


# ==========================================
# 3. APPLICATION HEADING
# ==========================================

st.title("🔊 Acoustic Machine Anomaly Detection System")

st.write(
    "Upload a machine audio recording to check whether "
    "the machine is operating normally or showing abnormal behaviour."
)

st.divider()


# ==========================================
# 4. SIDEBAR
# ==========================================

st.sidebar.title("About the System")

st.sidebar.write(
    "This application uses machine learning to detect "
    "anomalies in industrial machine sounds."
)

st.sidebar.write("Machine Type: Fan")
st.sidebar.write("Machine ID: id_00")
st.sidebar.write("Audio Format: WAV")
st.sidebar.write("Sample Rate: 16,000 Hz")
st.sidebar.write("Model: One-Class SVM")


# ==========================================
# 5. AUDIO UPLOAD
# ==========================================

st.subheader("Upload Audio")

uploaded_file = st.file_uploader(
    "Choose a WAV audio file",
    type=["wav"]
)


# ==========================================
# 6. FEATURE EXTRACTION AND PREDICTION
# ==========================================

if uploaded_file is not None:

    st.success("Audio file uploaded successfully!")

    st.write("File name:", uploaded_file.name)
    st.write("File size:", round(uploaded_file.size / 1024, 2), "KB")

    st.audio(uploaded_file.getvalue(), format="audio/wav")

    try:

        # Load audio at 16,000 Hz in mono
        audio, sr = librosa.load(
            io.BytesIO(uploaded_file.getvalue()),
            sr=16000,
            mono=True
        )

        if len(audio) == 0:
            raise ValueError("The uploaded audio file is empty.")

        # Extract the shared 34 features
        features = extract_features(audio, sr)

        features = np.asarray(features).flatten()

        st.subheader("Extracted Audio Features")

        st.write("Sample rate:", sr)
        st.write("Audio duration:", round(len(audio) / sr, 2), "seconds")
        st.write("Number of extracted features:", len(features))

        if len(features) != 34:

            st.error(
                f"Expected 34 features, but extracted {len(features)}. "
                "Please check feature_extraction.py."
            )

        else:

            st.success("Successfully extracted all 34 features!")

            # Create DataFrame using the exact saved feature order
            feature_df = pd.DataFrame(
                [features],
                columns=model_feature_names
            )

            st.dataframe(feature_df)

            # ==========================================
            # 7. FEATURE ORDER VERIFICATION
            # ==========================================

            scaler_feature_names = getattr(
                scaler, "feature_names_in_", None
            )

            if scaler_feature_names is not None:
                if list(scaler_feature_names) != list(model_feature_names):
                    raise ValueError(
                        "Feature order does not match the trained scaler."
                    )

            # ==========================================
            # 8. APPLY SAVED SCALER
            # ==========================================

            scaled_features = scaler.transform(feature_df)

            # ==========================================
            # 9. ONE-CLASS SVM PREDICTION
            # ==========================================

            decision_score = model.decision_function(scaled_features)

            # Aamna's notebook: anomaly score = negative decision score
            anomaly_score = -float(decision_score[0])

            # ==========================================
            # 10. APPLY SAVED ANOMALY THRESHOLD
            # ==========================================

            st.divider()
            st.subheader("AI Model Prediction")

            st.metric(
                "Anomaly Score",
                f"{anomaly_score:.6f}"
            )

            st.metric(
                "Anomaly Threshold",
                f"{threshold:.6f}"
            )

            if anomaly_score > threshold:

                st.error("🚨 ANOMALY DETECTED")

                st.write(
                    "The machine sound shows abnormal behaviour."
                )

            else:

                st.success("✅ NORMAL")

                st.write(
                    "The machine sound appears to be operating normally."
                )

    except Exception as e:

        st.error(f"Processing failed: {e}")

else:

    st.warning("Please upload a WAV audio file to continue.")


# ==========================================
# 11. FOOTER
# ==========================================

st.divider()

st.caption(
    "Acoustic Machine Anomaly Detection System | AI Project 2026"
)
