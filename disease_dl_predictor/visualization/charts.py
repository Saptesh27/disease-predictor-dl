"""Plotly chart builders for app visualization."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def disease_comparison_bar(bilstm_results: list, cnn_results: list) -> go.Figure:
    names = [r["disease"] for r in bilstm_results]
    bvals = [r["percentage"] for r in bilstm_results]
    cmap = {r["disease"]: r["percentage"] for r in cnn_results}
    cvals = [cmap.get(n, 0.0) for n in names]
    fig = go.Figure()
    fig.add_bar(name="BiLSTM", x=names, y=bvals, marker_color="#1f77b4", text=bvals, textposition="outside")
    fig.add_bar(name="CNN", x=names, y=cvals, marker_color="#ff7f0e", text=cvals, textposition="outside")
    fig.update_layout(barmode="group", title="Disease Prediction Comparison — BiLSTM vs CNN", yaxis_title="Probability %")
    return fig


def confidence_gauge(bilstm_conf: float, cnn_conf: float) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "indicator"}, {"type": "indicator"}]])
    for i, (name, val) in enumerate([("BiLSTM", bilstm_conf), ("CNN", cnn_conf)], start=1):
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=val,
                title={"text": name},
                gauge={"axis": {"range": [0, 100]}, "steps": [{"range": [0, 40], "color": "red"}, {"range": [40, 70], "color": "yellow"}, {"range": [70, 100], "color": "green"}]},
            ),
            row=1,
            col=i,
        )
    return fig


def _report_to_df(report: dict, label_encoder) -> pd.DataFrame:
    rows = []
    for key, metrics in report.items():
        if key in {"accuracy", "macro avg", "weighted avg"}:
            continue
        try:
            idx = int(float(key))
            disease = label_encoder.inverse_transform([idx])[0]
        except (TypeError, ValueError):
            disease = str(key)
        rows.append(
            {
                "disease": disease,
                "precision": float(metrics.get("precision", 0.0)) * 100,
                "recall": float(metrics.get("recall", 0.0)) * 100,
                "f1_score": float(metrics.get("f1-score", 0.0)) * 100,
                "support": int(metrics.get("support", 0)),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["disease", "precision", "recall", "f1_score", "support"])
    return pd.DataFrame(rows)


def confusion_matrix_chart(cm: list, class_names: list, model_name: str, top_n: int = 20) -> go.Figure:
    matrix = np.array(cm)
    support = matrix.sum(axis=1)
    top_indices = np.argsort(support)[::-1][:top_n]
    filtered_cm = matrix[np.ix_(top_indices, top_indices)]
    filtered_names = [class_names[i] for i in top_indices]
    row_sum = filtered_cm.sum(axis=1, keepdims=True)
    normalized = np.divide(filtered_cm.astype(float), row_sum, out=np.zeros_like(filtered_cm, dtype=float), where=row_sum != 0) * 100

    fig = go.Figure(
        data=go.Heatmap(
            z=normalized,
            x=filtered_names,
            y=filtered_names,
            text=filtered_cm,
            texttemplate="%{text}",
            colorscale="Blues",
            colorbar={"title": "% Correct"},
            hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{text}<br>Percentage: %{z:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Confusion Matrix - {model_name} (Top {len(filtered_names)} diseases shown)",
        xaxis_title="Predicted Disease",
        yaxis_title="Actual Disease",
        height=900,
    )
    return fig


def confusion_matrix_comparison(cm_bilstm: list, cm_cnn: list, class_names: list, top_n: int = 15) -> go.Figure:
    bilstm_m = np.array(cm_bilstm)
    cnn_m = np.array(cm_cnn)
    support = bilstm_m.sum(axis=1) + cnn_m.sum(axis=1)
    top_indices = np.argsort(support)[::-1][:top_n]
    b = bilstm_m[np.ix_(top_indices, top_indices)]
    c = cnn_m[np.ix_(top_indices, top_indices)]
    names = [class_names[i] for i in top_indices]
    b_norm = np.divide(b.astype(float), b.sum(axis=1, keepdims=True), out=np.zeros_like(b, dtype=float), where=b.sum(axis=1, keepdims=True) != 0) * 100
    c_norm = np.divide(c.astype(float), c.sum(axis=1, keepdims=True), out=np.zeros_like(c, dtype=float), where=c.sum(axis=1, keepdims=True) != 0) * 100

    fig = make_subplots(rows=1, cols=2, subplot_titles=("BiLSTM", "CNN"))
    fig.add_trace(
        go.Heatmap(z=b_norm, x=names, y=names, text=b, texttemplate="%{text}", colorscale="Blues", showscale=False),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(z=c_norm, x=names, y=names, text=c, texttemplate="%{text}", colorscale="Blues", colorbar={"title": "% Correct"}),
        row=1,
        col=2,
    )
    fig.update_layout(title="Confusion Matrix Comparison - BiLSTM vs CNN", height=700)
    return fig


def f1_precision_recall_bar(bilstm_report: dict, cnn_report: dict, label_encoder, top_n: int = 15) -> go.Figure:
    bdf = _report_to_df(bilstm_report, label_encoder).set_index("disease")
    cdf = _report_to_df(cnn_report, label_encoder).set_index("disease")
    combined = bdf.join(cdf, how="inner", lsuffix="_b", rsuffix="_c")
    combined["avg_f1"] = (combined["f1_score_b"] + combined["f1_score_c"]) / 2
    top = combined.sort_values("avg_f1", ascending=False).head(top_n)
    diseases = top.index.tolist()

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("F1 Score", "Precision", "Recall"),
    )
    metric_map = [("f1_score", 1), ("precision", 2), ("recall", 3)]
    for metric_name, row in metric_map:
        show_legend = row == 1
        fig.add_bar(
            x=diseases,
            y=top[f"{metric_name}_b"],
            name="BiLSTM",
            marker_color="#2196F3",
            text=np.round(top[f"{metric_name}_b"], 1),
            textposition="outside",
            showlegend=show_legend,
            row=row,
            col=1,
        )
        fig.add_bar(
            x=diseases,
            y=top[f"{metric_name}_c"],
            name="CNN",
            marker_color="#FF9800",
            text=np.round(top[f"{metric_name}_c"], 1),
            textposition="outside",
            showlegend=show_legend,
            row=row,
            col=1,
        )
    fig.update_layout(
        barmode="group",
        title=f"F1 Score, Precision & Recall - BiLSTM vs CNN (Top {len(diseases)} Diseases)",
        height=900,
    )
    fig.update_yaxes(range=[0, 105], gridcolor="rgba(150,150,150,0.25)")
    return fig


def overall_metrics_bar(bilstm_metrics: dict, cnn_metrics: dict) -> go.Figure:
    labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
    bvals = [bilstm_metrics["accuracy"], bilstm_metrics["precision"], bilstm_metrics["recall"], bilstm_metrics["f1"]]
    cvals = [cnn_metrics["accuracy"], cnn_metrics["precision"], cnn_metrics["recall"], cnn_metrics["f1"]]
    winner_is_bilstm = np.mean(bvals) >= np.mean(cvals)

    fig = go.Figure()
    fig.add_bar(
        name="BiLSTM",
        x=labels,
        y=bvals,
        marker={"color": "#2196F3", "line": {"color": "gold" if winner_is_bilstm else "#2196F3", "width": 3 if winner_is_bilstm else 1}},
        text=[f"{v:.1f}%" for v in bvals],
        textposition="outside",
    )
    fig.add_bar(
        name="CNN",
        x=labels,
        y=cvals,
        marker={"color": "#FF9800", "line": {"color": "gold" if not winner_is_bilstm else "#FF9800", "width": 3 if not winner_is_bilstm else 1}},
        text=[f"{v:.1f}%" for v in cvals],
        textposition="outside",
    )
    fig.add_hline(y=80, line_dash="dash", annotation_text="Good Threshold")
    fig.add_hline(y=90, line_dash="dash", annotation_text="Excellent Threshold")
    fig.update_layout(barmode="group", yaxis_range=[0, 105], title="Overall Model Performance Metrics")
    return fig


def precision_recall_per_class_heatmap(bilstm_report: dict, cnn_report: dict, label_encoder, top_n: int = 20) -> go.Figure:
    bdf = _report_to_df(bilstm_report, label_encoder).set_index("disease")
    cdf = _report_to_df(cnn_report, label_encoder).set_index("disease")
    top = bdf.sort_values("f1_score", ascending=False).head(top_n).index.tolist()
    matrix = np.array(
        [
            bdf.loc[top, "precision"].values,
            bdf.loc[top, "recall"].values,
            cdf.reindex(top)["precision"].fillna(0).values,
            cdf.reindex(top)["recall"].fillna(0).values,
        ]
    )
    rows = ["BiLSTM Precision", "BiLSTM Recall", "CNN Precision", "CNN Recall"]
    fig = ff.create_annotated_heatmap(
        z=np.round(matrix, 1),
        x=top,
        y=rows,
        colorscale="RdYlGn",
        showscale=True,
    )
    fig.update_layout(title=f"Precision & Recall Heatmap per Disease - Top {len(top)} Diseases", height=500)
    return fig


def f1_scatter_comparison(bilstm_report: dict, cnn_report: dict, label_encoder) -> go.Figure:
    bdf = _report_to_df(bilstm_report, label_encoder).set_index("disease")
    cdf = _report_to_df(cnn_report, label_encoder).set_index("disease")
    merged = bdf.join(cdf, how="inner", lsuffix="_b", rsuffix="_c").reset_index()
    merged["winner"] = np.where(
        merged["f1_score_b"] > merged["f1_score_c"],
        "BiLSTM",
        np.where(merged["f1_score_b"] < merged["f1_score_c"], "CNN", "Tied"),
    )

    fig = px.scatter(
        merged,
        x="f1_score_b",
        y="f1_score_c",
        size="support_b",
        color="winner",
        color_discrete_map={"BiLSTM": "#2196F3", "CNN": "#FF9800", "Tied": "gray"},
        hover_name="disease",
        labels={"f1_score_b": "BiLSTM F1 (%)", "f1_score_c": "CNN F1 (%)"},
    )
    fig.update_traces(
        hovertemplate="Disease: %{hovertext}<br>BiLSTM F1: %{x:.2f}%<br>CNN F1: %{y:.2f}%<br>Samples: %{marker.size}<extra></extra>"
    )
    fig.add_shape(type="line", x0=0, y0=0, x1=100, y1=100, line={"dash": "dash", "color": "black"})
    fig.add_annotation(x=95, y=95, text="Equal Performance", showarrow=False)
    merged["delta"] = (merged["f1_score_b"] - merged["f1_score_c"]).abs()
    for _, row in merged.nlargest(5, "delta").iterrows():
        fig.add_annotation(x=row["f1_score_b"], y=row["f1_score_c"], text=row["disease"], showarrow=True, arrowhead=1)
    fig.update_layout(title="Per-Disease F1 Score - BiLSTM vs CNN Scatter")
    return fig


def training_loss_curves(bilstm_history: dict, cnn_history: dict) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(y=bilstm_history.get("loss", []), mode="lines+markers", name="BiLSTM Train Loss")
    fig.add_scatter(y=bilstm_history.get("val_loss", []), mode="lines+markers", name="BiLSTM Val Loss", line={"dash": "dash"})
    fig.add_scatter(y=cnn_history.get("loss", []), mode="lines+markers", name="CNN Train Loss")
    fig.add_scatter(y=cnn_history.get("val_loss", []), mode="lines+markers", name="CNN Val Loss", line={"dash": "dash"})
    fig.update_layout(title="Training vs Validation Loss", xaxis_title="Epoch", yaxis_title="Loss")
    return fig


def model_metrics_radar(bilstm_metrics: dict, cnn_metrics: dict) -> go.Figure:
    categories = ["accuracy", "precision", "recall", "f1"]
    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=[bilstm_metrics[m] for m in categories] + [bilstm_metrics[categories[0]]],
            theta=[m.title() for m in categories] + [categories[0].title()],
            fill="toself",
            name="BiLSTM",
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=[cnn_metrics[m] for m in categories] + [cnn_metrics[categories[0]]],
            theta=[m.title() for m in categories] + [categories[0].title()],
            fill="toself",
            name="CNN",
        )
    )
    fig.update_layout(title="Model Metrics Radar", polar={"radialaxis": {"visible": True, "range": [0, 100]}})
    return fig
