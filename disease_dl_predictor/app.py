"""Streamlit app for disease prediction comparison."""

import pickle

import streamlit as st

from config.settings import settings
from models.model_manager import ModelManager
from training.evaluator import Evaluator
from visualization.charts import (
    confidence_gauge,
    confusion_matrix_chart,
    confusion_matrix_comparison,
    disease_comparison_bar,
    f1_precision_recall_bar,
    f1_scatter_comparison,
    model_metrics_radar,
    overall_metrics_bar,
    precision_recall_per_class_heatmap,
    training_loss_curves,
)

st.set_page_config(page_title="Disease Predictor - BiLSTM vs CNN", page_icon=":hospital:", layout="wide")
st.title("Disease Predictor")
st.caption("Deep Learning Analysis - BiLSTM vs CNN")


@st.cache_resource
def load_model_manager():
    manager = ModelManager()
    manager.load_all()
    return manager


@st.cache_resource
def load_eval_results():
    try:
        with open(settings.EVAL_RESULTS_PATH, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None
    except Exception:
        return None


@st.cache_resource
def load_training_history():
    try:
        with open(settings.TRAINING_HISTORY_PATH, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None
    except Exception:
        return None


try:
    with st.spinner("Loading AI models..."):
        model_manager = load_model_manager()
except FileNotFoundError:
    st.error("Models not trained yet. Please run: python train.py")
    st.stop()

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

st.divider()
st.markdown("## Model Evaluation Dashboard")
st.caption("Results from training evaluation on test dataset")

eval_results = load_eval_results()
if eval_results is None:
    st.warning("No evaluation data found. Please run: python train.py")
else:
    class_names = list(model_manager.label_encoder.classes_)
    bilstm_metrics = eval_results["bilstm"]
    cnn_metrics = eval_results["cnn"]

    col1, col2, col3, col4 = st.columns(4)
    bilstm_acc_delta = bilstm_metrics["accuracy"] - cnn_metrics["accuracy"]
    bilstm_f1_delta = bilstm_metrics["f1"] - cnn_metrics["f1"]
    cnn_acc_delta = cnn_metrics["accuracy"] - bilstm_metrics["accuracy"]
    cnn_f1_delta = cnn_metrics["f1"] - bilstm_metrics["f1"]
    col1.metric("BiLSTM Accuracy", f"{bilstm_metrics['accuracy']:.2f}%", delta=f"{bilstm_acc_delta:+.2f}% vs CNN")
    col2.metric("BiLSTM F1 Score", f"{bilstm_metrics['f1']:.2f}%", delta=f"{bilstm_f1_delta:+.2f}% vs CNN")
    col3.metric("CNN Accuracy", f"{cnn_metrics['accuracy']:.2f}%", delta=f"{cnn_acc_delta:+.2f}% vs BiLSTM")
    col4.metric("CNN F1 Score", f"{cnn_metrics['f1']:.2f}%", delta=f"{cnn_f1_delta:+.2f}% vs BiLSTM")

    st.plotly_chart(overall_metrics_bar(bilstm_metrics, cnn_metrics), use_container_width=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Confusion Matrix",
            "F1 / Precision / Recall",
            "Per-Class Heatmap",
            "F1 Scatter Plot",
            "Training History",
        ]
    )

    with tab1:
        top_n = st.slider("Show top N diseases", 5, 30, 15)
        sub_col1, sub_col2 = st.columns(2)
        with sub_col1:
            st.markdown("### BiLSTM")
            st.plotly_chart(
                confusion_matrix_chart(eval_results["bilstm"]["confusion_matrix"], class_names, "BiLSTM", top_n=top_n),
                use_container_width=True,
            )
        with sub_col2:
            st.markdown("### CNN")
            st.plotly_chart(
                confusion_matrix_chart(eval_results["cnn"]["confusion_matrix"], class_names, "CNN", top_n=top_n),
                use_container_width=True,
            )
        st.markdown("### Side-by-Side Comparison")
        st.plotly_chart(
            confusion_matrix_comparison(
                eval_results["bilstm"]["confusion_matrix"],
                eval_results["cnn"]["confusion_matrix"],
                class_names,
                top_n=top_n,
            ),
            use_container_width=True,
        )

    with tab2:
        st.plotly_chart(
            f1_precision_recall_bar(eval_results["bilstm_report"], eval_results["cnn_report"], model_manager.label_encoder),
            use_container_width=True,
        )
        st.markdown("### Full Per-Disease Metrics Table")
        model_choice = st.radio("Select model", ["BiLSTM", "CNN"], horizontal=True)
        selected_report = eval_results["bilstm_report"] if model_choice == "BiLSTM" else eval_results["cnn_report"]
        df = Evaluator.get_per_class_df(selected_report, model_manager.label_encoder)
        filter_disease = st.text_input("Search disease")
        if filter_disease:
            df = df[df["disease"].str.contains(filter_disease, case=False, na=False)]
        st.dataframe(df.style.background_gradient(subset=["f1_score"], cmap="RdYlGn"), use_container_width=True, height=420)
        st.download_button(
            "Download as CSV",
            data=df.to_csv(index=False),
            file_name=f"{model_choice.lower()}_metrics.csv",
            mime="text/csv",
        )

    with tab3:
        top_n_heat = st.slider("Diseases to show in heatmap", 10, 30, 20, key="heat_slider")
        st.plotly_chart(
            precision_recall_per_class_heatmap(
                eval_results["bilstm_report"],
                eval_results["cnn_report"],
                model_manager.label_encoder,
                top_n=top_n_heat,
            ),
            use_container_width=True,
        )

    with tab4:
        st.plotly_chart(
            f1_scatter_comparison(eval_results["bilstm_report"], eval_results["cnn_report"], model_manager.label_encoder),
            use_container_width=True,
        )

    with tab5:
        history = load_training_history()
        if history is None:
            st.info("No training history found. Re-run training to generate history plots.")
        else:
            st.plotly_chart(training_loss_curves(history.get("bilstm", {}), history.get("cnn", {})), use_container_width=True)
            st.plotly_chart(model_metrics_radar(bilstm_metrics, cnn_metrics), use_container_width=True)
