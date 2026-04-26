"""Plotly chart builders for app visualization."""

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
