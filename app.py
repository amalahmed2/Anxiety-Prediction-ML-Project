import streamlit as st
import pandas as pd
import joblib

# Load trained models
xgb_model = joblib.load('xgb_model.pkl')
rf_model = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.pkl')

# Model descriptions
model_descriptions = {
    "XGBoost": "✅ Highest accuracy and excellent performance in predicting anxiety levels.",
    "Random Forest": "🌿 Fast and interpretable model with good overall accuracy."
}

# App title
st.title("💬 Anxiety Prediction Web App")

# Updated model info title
st.markdown("## 🔍 Which Model is Right for You?")
for name, desc in model_descriptions.items():
    st.markdown(f"### {name}\n{desc}")

# Model selection
st.markdown("## 🔍 Select a Model")
model_choice = st.selectbox("Choose a model to use:", ["XGBoost", "Random Forest"])

# Upload data
st.markdown("## 📤 Upload a CSV File")
uploaded_file = st.file_uploader("Upload your test data (.csv)", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        # ✅ Check for matching columns regardless of order
        expected_cols = ['Age', 'SWL_T', 'SPIN_T', 'Narcissism',
                         'Gender_encoded', 'Education_encoded',
                         'Country_encoded', 'Value', 'Subgroup_encoded']

        if sorted(df.columns) != sorted(expected_cols):
            st.error("❌ Uploaded file doesn't match required feature structure.")
            st.stop()

        st.write("### 🔎 Uploaded Data Preview")
        st.dataframe(df.head())

        # Preprocessing
        X_scaled = scaler.transform(df)

        # Select model
        model = xgb_model if model_choice == "XGBoost" else rf_model

        # Make predictions
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)

        # Display results
        st.markdown("## 📈 Prediction Results")
        df["Predicted Class"] = predictions
        df["Anxiety Probability (%)"] = (probabilities[:, 1] * 100).round(2)
        st.dataframe(df)

    except Exception as e:
        st.error(f"❌ Error: {e}")
