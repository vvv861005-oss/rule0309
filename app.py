import os
import sys
import time
import base64
import threading
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import cv2
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

# Core Logic Extraction
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from engine import RevolutionMaster

# ──────────────────────────────────────────────────────────────────────────────
# Master Control Singleton
# ──────────────────────────────────────────────────────────────────────────────

if not hasattr(dash, "_REVOLATION_SERVER"):
    server = RevolutionMaster()
    server.run()
    dash._REVOLATION_SERVER = server

master = dash._REVOLATION_SERVER

# ──────────────────────────────────────────────────────────────────────────────
# Dash App Definition
# ──────────────────────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG, "https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Share+Tech+Mono&display=swap"],
    title="REVOLUTION HYPERVISION — DASH PRO"
)

# ──────────────────────────────────────────────────────────────────────────────
# Layout Blocks
# ──────────────────────────────────────────────────────────────────────────────

def create_indicator(label, value_id, color="#00e5ff"):
    return dbc.Col([
        html.Div([
            html.Div(label, style={"color": "#88aaff", "font-size": "0.75rem", "text-transform": "uppercase", "letter-spacing": "2px"}),
            html.Div("0.00", id=value_id, style={"color": color, "font-size": "2rem", "font-family": "'Share Tech Mono', monospace", "text-shadow": f"0 0 10px {color}80"})
        ], className="p-3 mb-3", style={"background": "rgba(10,10,30,0.6)", "border-left": f"4px solid {color}", "border-radius": "4px"})
    ], width=6, lg=3)

app.layout = html.Div([
    # Cyber Header
    html.Header([
        dbc.Row([
            dbc.Col([
                html.H1("▼ REVOLUTION HYPERVISION", style={"font-family": "'Orbitron', sans-serif", "color": "#00e5ff", "margin": 0, "letter-spacing": "5px"}),
                html.P("Quantum-RF Fusion Core | Observational Saturation Engine v5.0", style={"color": "#7b1fa2", "font-family": "'Share Tech Mono', monospace", "margin": 0})
            ], width=8),
            dbc.Col([
                html.Div([
                    html.Span("● SYSTEM STABLE", style={"color": "#76ff03", "font-family": "'Orbitron', sans-serif", "font-size": "0.8rem", "margin-right": "15px"}),
                    dbc.Button("RE-CALIBRATE", id="btn-recal", color="info", outline=True, size="sm")
                ], style={"float": "right", "margin-top": "15px"})
            ], width=4)
        ])
    ], style={"background": "#050510", "border-bottom": "2px solid #00e5ff", "padding": "20px 40px", "box-shadow": "0 0 20px rgba(0, 229, 255, 0.2)"}),

    dbc.Container([
        # KPIs
        dbc.Row([
            create_indicator("Latent Energy", "stat-energy"),
            create_indicator("Respiration Phase", "stat-resp", color="#7b1fa2"),
            create_indicator("Heart Oscillations", "stat-heart", color="#ff1744"),
            create_indicator("System Confidence", "stat-conf", color="#76ff03"),
        ], className="mt-4"),

        # Main Visualization
        dbc.Row([
            # 3D Fusion (Left Large)
            dbc.Col([
                html.Div([
                    html.H5("FUSED SPATIAL LATTICE (RF × DEPTH × POSE)", style={"color": "#88aaff", "font-family": "'Orbitron', sans-serif", "margin-bottom": "15px"}),
                    dcc.Graph(id="graph-3d", style={"height": "650px"}, config={"displayModeBar": False})
                ], style={"background": "rgba(5,5,15,0.8)", "border": "1px solid #333", "border-radius": "8px", "padding": "20px"})
            ], width=12, lg=8),

            # Status Feed (Right)
            dbc.Col([
                html.Div([
                    html.H5("LIVE OPTICAL TELEMETRY", style={"color": "#88aaff", "font-family": "'Orbitron', sans-serif"}),
                    html.Img(id="feed-optical", style={"width": "100%", "border": "1px solid #00e5ff", "border-radius": "4px", "margin-bottom": "20px"}),
                    
                    html.H5("PREDICTED BEHAVIOR", style={"color": "#88aaff", "font-family": "'Orbitron', sans-serif"}),
                    html.Div("STILL / MONITORING", id="text-action", style={"background": "#001a1a", "color": "#76ff03", "padding": "15px", "font-size": "1.5rem", "text-align": "center", "font-family": "'Share Tech Mono', monospace", "border": "1px dashed #76ff03", "border-radius": "4px"}),
                    
                    html.Div([
                        html.P("Coordinate Mapping (Fused):", style={"color": "#88aaff", "margin": "20px 0 5px 0"}),
                        html.Code("X: 0.00 | Y: 0.00 | Z: 0.00", id="coord-fused", style={"color": "#00e5ff", "font-size": "1rem"})
                    ])
                ], style={"background": "rgba(5,5,15,0.8)", "border": "1px solid #333", "border-radius": "8px", "padding": "20px", "height": "100%"})
            ], width=12, lg=4)
        ], className="mt-4"),

        # Signal Analytics
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H6("RF DYNAMICS (RSSI/SNR TREND)", style={"color": "#88aaff", "margin-bottom": "15px"}),
                    dcc.Graph(id="graph-signal", style={"height": "250px"}, config={"displayModeBar": False})
                ], style={"background": "rgba(5,5,15,0.8)", "border": "1px solid #333", "border-radius": "8px", "padding": "15px"})
            ], width=12)
        ], className="my-4")
    ], fluid=True),

    dcc.Interval(id="clk-pulse", interval=400, n_intervals=0)
], style={"background": "#020205", "min-height": "100vh"})

# ──────────────────────────────────────────────────────────────────────────────
# Pulse Callbacks
# ──────────────────────────────────────────────────────────────────────────────

@app.callback(
    [Output("stat-energy", "children"), Output("stat-resp", "children"), Output("stat-heart", "children"), Output("stat-conf", "children"),
     Output("feed-optical", "src"), Output("text-action", "children"), Output("coord-fused", "children"),
     Output("graph-3d", "figure"), Output("graph-signal", "figure")],
    [Input("clk-pulse", "n_intervals")]
)
def system_pulse(n):
    state = master.get_state()
    
    # 1. Metric update
    met = state["metrics"]
    energy = f"{met['energy']:.4f}"
    resp = f"{met['resp']:.1f} bpm"
    heart = f"{met['heart']:.0f} bpm"
    conf = f"{met['conf']*100:.1f}%"

    # 2. Vision update
    vis = state["vision"]
    rgb = vis.get("rgb")
    feed_src = ""
    if rgb is not None:
        rgb_small = cv2.resize(rgb, (400, 300))
        # Draw simulated pose overlay in app for visual flair
        for i, lm in enumerate(vis.get("lms", [])):
            px, py = int(lm.x * 400), int(lm.y * 300)
            cv2.circle(rgb_small, (px, py), 3, (0, 229, 255), -1)
        
        _, buf = cv2.imencode('.jpg', cv2.cvtColor(rgb_small, cv2.COLOR_RGB2BGR))
        feed_src = f"data:image/jpeg;base64,{base64.b64encode(buf).decode()}"
    
    action = f"{vis.get('action', 'STILL')} / DETECTED"
    fused_xyz = vis.get("fused_xyz", (0,0,0))
    coord_txt = f"X: {fused_xyz[0]:+06.2f} | Y: {fused_xyz[1]:+06.2f} | Z: {fused_xyz[2]:+06.2f}"

    # 3. 3D Visualization
    fus = state["fusion"]
    vox = fus.get("voxels", [])
    fig_3d = go.Figure()

    # Lattice Cloud
    if vox:
        fig_3d.add_trace(go.Scatter3d(
            x=[v["x"] for v in vox], y=[v["y"] for v in vox], z=[v["z"] for v in vox],
            mode="markers", marker=dict(size=4, color=[v["p"] for v in vox], colorscale="Cividis", opacity=0.3, cmin=0, cmax=1),
            name="RF LATTICE"
        ))

    # Pose/Skeleton in 3D
    lms = vis.get("lms", [])
    if lms:
        fig_3d.add_trace(go.Scatter3d(
            x=[l.wx for l in lms], y=[l.wz for l in lms], z=[l.wy for l in lms],
            mode="markers+lines", marker=dict(size=4, color="#00ff88"),
            line=dict(color="#00ffcc", width=1), name="ML POSE"
        ))

    # Trajectory
    traj = vis.get("traj", [])
    if len(traj) > 1:
        fig_3d.add_trace(go.Scatter3d(
            x=[p[0] for p in traj], y=[p[2] for p in traj], z=[p[1] for p in traj],
            mode="lines", line=dict(color="yellow", width=2), name="PATH"
        ))

    fig_3d.update_layout(
        scene=dict(
            xaxis=dict(range=[0, 5], backgroundcolor="#050510", color="white", gridcolor="#222", zeroline=False),
            yaxis=dict(range=[0, 5], backgroundcolor="#050510", color="white", gridcolor="#222", zeroline=False),
            zaxis=dict(range=[0, 3], backgroundcolor="#050510", color="white", gridcolor="#222", zeroline=False),
            aspectratio=dict(x=1, y=1, z=0.6)
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(x=0, y=1, font=dict(color="white"))
    )

    # 4. Signal Graph
    rf = state["rf"]
    fig_sig = go.Figure()
    fig_sig.add_trace(go.Scatter(y=rf.get("rssi", []), name="RSSI", line=dict(color="#ff6b6b", width=2)))
    fig_sig.add_trace(go.Scatter(y=rf.get("snr", []), name="SNR", line=dict(color="#4ecdc4", width=2)))
    fig_sig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0,r=0,t=0,b=0), font=dict(color="white"),
        xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="#222")
    )

    return energy, resp, heart, conf, feed_src, action, coord_txt, fig_3d, fig_sig

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8050, debug=False)
