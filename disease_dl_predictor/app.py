"""Streamlit app for disease prediction comparison."""

import streamlit as st

from models.model_manager import ModelManager
from visualization.charts import confidence_gauge, disease_comparison_bar

st.set_page_config(page_title="Disease Predictor - BiLSTM vs CNN", page_icon=":hospital:", layout="wide")
st.title("Disease Predictor")
st.caption("Deep Learning Analysis - BiLSTM vs CNN")


@st.cache_resource
def load_model_manager():
    manager = ModelManager()
    manager.load_all()
    return manager


try:
    with st.spinner("Loading AI models..."):
        model_manager = load_model_manager()
except FileNotFoundError:
    st.error("Models not trained yet. Please run: python train.py")
    st.stop()

tab_symptoms, tab_prescription = st.tabs(["Symptom Analysis", "Prescription Analysis"])

with tab_symptoms:
    symptom_text = st.text_area("Describe symptoms", height=160, key="symptom_input")
    if st.button("Analyze Symptoms", key="analyze_symptoms_button"):
        result = model_manager.predict_both(symptom_text, "symptoms")
        st.write(f"BiLSTM: {result['bilstm']['top_disease']} ({result['bilstm']['top_confidence']}%)")
        st.write(f"CNN: {result['cnn']['top_disease']} ({result['cnn']['top_confidence']}%)")
        st.plotly_chart(
            disease_comparison_bar(result["bilstm"]["predictions"], result["cnn"]["predictions"]),
            use_container_width=True,
        )
        st.plotly_chart(
            confidence_gauge(result["bilstm"]["top_confidence"], result["cnn"]["top_confidence"]),
            use_container_width=True,
        )

with tab_prescription:
    prescription_text = st.text_area("Paste prescription text", height=160, key="prescription_input")
    if st.button("Analyze Prescription", key="analyze_prescription_button"):
        result = model_manager.predict_both(prescription_text, "prescription")
        st.write(f"{result['winner']} is more confident.")
