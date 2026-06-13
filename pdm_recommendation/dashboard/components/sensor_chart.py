"""Komponen chart kondisi sensor dengan Plotly."""
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

def render_sensor_chart(machine_data: pd.DataFrame, sensor_cols: list, last_n_hours: int = 168):
    """
    Render line chart 4 sensor (7 hari terakhir by default).
    
    Args:
        machine_data: DataFrame satu mesin, sudah sorted by datetime
        sensor_cols:  list nama kolom sensor
        last_n_hours: berapa jam terakhir yang ditampilkan
    """
    data = machine_data.tail(last_n_hours).copy()
    if data.empty:
        st.warning("Tidak ada data sensor untuk ditampilkan.")
        return

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[c.capitalize() for c in sensor_cols],
        shared_xaxes=True,
        vertical_spacing=0.12,
    )

    positions = [(1,1), (1,2), (2,1), (2,2)]
    colors    = ["#3B82F6", "#10B981", "#F59E0B", "#EF4444"]

    for i, (col, (row, col_pos), color) in enumerate(zip(sensor_cols, positions, colors)):
        if col not in data.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=data["datetime"], y=data[col],
                mode="lines", name=col,
                line=dict(color=color, width=1.5),
                showlegend=False,
            ),
            row=row, col=col_pos,
        )

    fig.update_layout(
        height=420,
        margin=dict(l=0, r=0, t=40, b=0),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)
