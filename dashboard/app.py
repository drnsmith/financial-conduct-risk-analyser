"""
Financial Conduct Risk Analyser — Dashboard
============================================
Combines quantitative financial scoring (Altman Z, Beneish M, Piotroski F)
with qualitative document risk extraction (Claude-powered) into a unified
regulatory risk assessment platform.
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(override=True)

import plotly.graph_objects as go
import plotly.express as px
import dash
from dash import dcc, html, Input, Output, callback, dash_table
import dash_bootstrap_components as dbc

ROOT      = Path(__file__).resolve().parents[1]
DATA_PROC = ROOT / "data" / "processed"

# ── DESIGN ────────────────────────────────────────────────────────────────────
DARK_BG   = "#060d1f"
CARD_BG   = "#0d1730"
BORDER    = "#1a2744"
ACCENT    = "#00c9a7"
ACCENT2   = "#4361ee"
ACCENT3   = "#f72585"
WARN      = "#ffd166"
TEXT      = "#e2e8f0"
TEXT_DIM  = "#64748b"
FONT_BODY = "'IBM Plex Sans', sans-serif"
FONT_MONO = "'IBM Plex Mono', monospace"

RISK_COLORS = {
    "High":    "#ef4444",
    "Medium":  "#f59e0b",
    "Low":     "#10b981",
    "Unknown": "#64748b",
}

CATEGORY_COLORS = {
    "conduct_risk":     "#f72585",
    "model_risk":       "#4361ee",
    "operational_risk": "#ffd166",
    "financial_risk":   "#ef4444",
    "regulatory_risk":  "#fb5607",
    "systemic_risk":    "#8338ec",
}

LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family=FONT_BODY, color=TEXT),
    margin=dict(l=20, r=20, t=40, b=20),
)
AXIS = dict(gridcolor=BORDER, linecolor=BORDER, zerolinecolor=BORDER)

# ── DATA ──────────────────────────────────────────────────────────────────────
def load_screen_results() -> pd.DataFrame:
    path = DATA_PROC / "screen_results.parquet"
    if path.exists():
        return pd.read_parquet(path)
    # Auto-run screener if no data
    try:
        import sys
        sys.path.insert(0, str(ROOT))
        from src.quantitative.screener import screen_companies
        import logging
        logging.basicConfig(level=logging.WARNING)
        print("No screen data found — running screener...")
        df = screen_companies(
            ["WOW.AX","BHP.AX","CSL.AX","WES.AX","CBA.AX","NAB.AX",
             "ANZ.AX","TLS.AX","AIR.NZ","FPH.NZ","ATM.NZ","RYM.NZ",
             "FBU.NZ","MFT.NZ","AIA.NZ","EBO.NZ"], "ASX"
        )
        DATA_PROC.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        return df
    except Exception as e:
        print(f"Screener failed: {e}")
        return pd.DataFrame()

def load_document_analysis() -> list:
    path = DATA_PROC / "document_analysis.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return []

SCREEN  = load_screen_results()
DOC_ANALYSIS = load_document_analysis()

# Flatten all signals
ALL_SIGNALS = []
for doc in DOC_ANALYSIS:
    for sig in doc.get("risk_signals", []):
        sig["document"] = doc["document"]
        sig["document_type"] = doc["document_type"]
        ALL_SIGNALS.append(sig)
SIGNALS_DF = pd.DataFrame(ALL_SIGNALS) if ALL_SIGNALS else pd.DataFrame()

# ── FIGURES ───────────────────────────────────────────────────────────────────
def fig_risk_tier_distribution():
    if SCREEN.empty:
        return go.Figure()
    counts = SCREEN["risk_tier"].value_counts()
    colors = [RISK_COLORS.get(t, "#64748b") for t in counts.index]
    fig = go.Figure(go.Bar(
        x=counts.index, y=counts.values,
        marker_color=colors,
        text=counts.values, textposition="outside",
        textfont=dict(color=TEXT_DIM, family=FONT_MONO),
    ))
    fig.update_layout(**LAYOUT,
        title=dict(text="Companies by Risk Tier", font=dict(size=13, color=TEXT_DIM)),
        xaxis=dict(**AXIS), yaxis=dict(**AXIS), height=300)
    return fig


def fig_score_scatter():
    if SCREEN.empty:
        return go.Figure()
    df = SCREEN.dropna(subset=["altman_z", "piotroski_f"])
    if df.empty:
        return go.Figure()
    colors = [RISK_COLORS.get(t, "#64748b") for t in df["risk_tier"]]
    fig = go.Figure(go.Scatter(
        x=df["altman_z"], y=df["piotroski_f"],
        mode="markers+text",
        text=df["ticker"],
        textposition="top center",
        textfont=dict(size=9, color=TEXT_DIM),
        marker=dict(size=10, color=colors, line=dict(color=BORDER, width=1)),
        hovertemplate="<b>%{text}</b><br>Altman Z: %{x:.2f}<br>Piotroski F: %{y}<extra></extra>",
    ))
    fig.add_vline(x=1.81, line_dash="dot", line_color=RISK_COLORS["High"],
                  annotation_text="Distress threshold",
                  annotation_font=dict(color=RISK_COLORS["High"], size=10))
    fig.add_vline(x=2.99, line_dash="dot", line_color=RISK_COLORS["Low"],
                  annotation_text="Safe threshold",
                  annotation_font=dict(color=RISK_COLORS["Low"], size=10))
    fig.update_layout(**LAYOUT,
        title=dict(text="Altman Z vs Piotroski F — Company Risk Map", font=dict(size=13, color=TEXT_DIM)),
        xaxis=dict(**AXIS, title="Altman Z-score"),
        yaxis=dict(**AXIS, title="Piotroski F-score"),
        height=420)
    return fig


def fig_document_signals():
    if SIGNALS_DF.empty:
        return go.Figure()
    breakdown = SIGNALS_DF.groupby(["document","risk_category"]).size().reset_index(name="count")
    fig = px.bar(breakdown, x="document", y="count", color="risk_category",
                 color_discrete_map=CATEGORY_COLORS,
                 barmode="stack")
    fig.update_layout(**LAYOUT,
        title=dict(text="Risk Signals by Document and Category", font=dict(size=13, color=TEXT_DIM)),
        xaxis=dict(**AXIS, title=""),
        yaxis=dict(**AXIS, title="Signal count"),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=BORDER),
        height=350)
    return fig


def fig_severity_breakdown():
    if SIGNALS_DF.empty:
        return go.Figure()
    counts = SIGNALS_DF["severity"].value_counts()
    colors = [RISK_COLORS.get(s, "#64748b") for s in counts.index]
    fig = go.Figure(go.Pie(
        labels=counts.index, values=counts.values,
        marker=dict(colors=colors, line=dict(color=BORDER, width=2)),
        textfont=dict(color=TEXT),
        hole=0.5,
    ))
    fig.update_layout(**LAYOUT,
        title=dict(text="Signal Severity Distribution", font=dict(size=13, color=TEXT_DIM)),
        height=300)
    return fig


# ── LAYOUT ────────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.CYBORG,
        "https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap",
    ],
    title="Financial Conduct Risk Analyser",
    suppress_callback_exceptions=True,
)
server = app.server


def kpi(value, label, color=ACCENT):
    return html.Div([
        html.Div(str(value), style={
            "fontFamily": FONT_MONO, "fontSize": "20px",
            "fontWeight": "700", "color": color,
        }),
        html.Div(label, style={
            "fontSize": "9px", "color": TEXT_DIM,
            "letterSpacing": "1px", "textTransform": "uppercase",
        }),
    ], style={
        "background": CARD_BG, "border": f"1px solid {BORDER}",
        "borderRadius": "8px", "padding": "12px 16px", "textAlign": "center",
    })


HEADER = html.Div([
    html.Div([
        html.Div([
            html.Span("FINANCIAL CONDUCT RISK · NZX / ASX", style={
                "fontFamily": FONT_MONO, "fontSize": "10px",
                "letterSpacing": "3px", "color": ACCENT, "fontWeight": "600",
            }),
            html.H1("Financial Conduct Risk Analyser", style={
                "fontFamily": FONT_BODY, "fontWeight": "700",
                "fontSize": "clamp(18px, 2.5vw, 28px)",
                "color": TEXT, "margin": "6px 0 4px",
            }),
            html.P(
                "Quantitative financial scoring · Document risk extraction · Valimetrica validity assessment",
                style={"color": TEXT_DIM, "fontSize": "12px", "margin": "0"}
            ),
        ]),
        html.Div([
            kpi(len(SCREEN), "Companies screened"),
            kpi(
                len(SCREEN[SCREEN["risk_tier"]=="High"]) if not SCREEN.empty else 0,
                "High risk", color=RISK_COLORS["High"]
            ),
            kpi(len(DOC_ANALYSIS), "Documents analysed"),
            kpi(len(ALL_SIGNALS), "Risk signals extracted"),
        ], style={"display": "flex", "gap": "10px", "flexWrap": "wrap"}),
    ], style={
        "display": "flex", "justifyContent": "space-between",
        "alignItems": "center", "flexWrap": "wrap", "gap": "16px",
        "padding": "24px 32px",
        "borderBottom": f"1px solid {BORDER}",
        "background": f"linear-gradient(135deg, {DARK_BG} 0%, #080f24 100%)",
    }),
])

TABS = dbc.Tabs([
    dbc.Tab(label="Company Screener",    tab_id="tab-screen"),
    dbc.Tab(label="Risk Map",            tab_id="tab-map"),
    dbc.Tab(label="Document Analysis",   tab_id="tab-docs"),
    dbc.Tab(label="Risk Signals",        tab_id="tab-signals"),
    dbc.Tab(label="Validity Assessment", tab_id="tab-validity"),
], id="main-tabs", active_tab="tab-screen", style={
    "padding": "0 32px",
    "borderBottom": f"1px solid {BORDER}",
    "background": DARK_BG,
})

app.layout = html.Div([
    HEADER, TABS,
    html.Div(id="tab-content", style={"padding": "24px 32px"}),
], style={"background": DARK_BG, "minHeight": "100vh", "fontFamily": FONT_BODY})


# ── CALLBACKS ─────────────────────────────────────────────────────────────────
@callback(Output("tab-content", "children"), Input("main-tabs", "active_tab"))
def render_tab(tab):

    if tab == "tab-screen":
        if SCREEN.empty:
            return html.P("No screen results. Run src/quantitative/screener.py first.",
                         style={"color": TEXT_DIM})
        cols = ["ticker","company","exchange","risk_tier","altman_zone","beneish_flag","piotroski_tier","risk_drivers"]
        cols = [c for c in cols if c in SCREEN.columns]
        return html.Div([
            dcc.Graph(figure=fig_risk_tier_distribution()),
            dash_table.DataTable(
                data=SCREEN[cols].to_dict("records"),
                columns=[{"name": c.replace("_"," ").title(), "id": c} for c in cols],
                style_table={"overflowX": "auto"},
                style_cell={
                    "backgroundColor": CARD_BG, "color": TEXT,
                    "border": f"1px solid {BORDER}",
                    "fontFamily": FONT_BODY, "fontSize": "12px",
                    "textAlign": "left", "padding": "8px 12px",
                },
                style_header={
                    "backgroundColor": DARK_BG, "color": ACCENT,
                    "fontFamily": FONT_MONO, "fontSize": "11px",
                    "border": f"1px solid {BORDER}",
                },
                style_data_conditional=[
                    {"if": {"filter_query": '{risk_tier} = "High"'},
                     "backgroundColor": "#2d0a0a", "color": RISK_COLORS["High"]},
                    {"if": {"filter_query": '{risk_tier} = "Medium"'},
                     "backgroundColor": "#2d1a00"},
                ],
                page_size=20, sort_action="native", filter_action="native",
            )
        ])

    elif tab == "tab-map":
        return dcc.Graph(figure=fig_score_scatter())

    elif tab == "tab-docs":
        if not DOC_ANALYSIS:
            return html.P("No document analysis results.", style={"color": TEXT_DIM})
        return html.Div([
            dcc.Graph(figure=fig_document_signals()),
            html.Div([
                html.Div([
                    html.Div(doc["document_type"], style={
                        "fontFamily": FONT_MONO, "fontSize": "10px",
                        "color": ACCENT, "letterSpacing": "1px",
                        "marginBottom": "6px",
                    }),
                    html.H4(doc["document"].replace("_"," ").title(), style={
                        "color": TEXT, "fontSize": "14px", "margin": "0 0 8px",
                    }),
                    html.P(doc.get("summary",""), style={
                        "color": TEXT_DIM, "fontSize": "12px", "margin": "0 0 10px",
                    }),
                    html.Div([
                        html.Span(f"{doc['total_signals']} signals",
                                 style={"color": ACCENT, "fontFamily": FONT_MONO,
                                        "fontSize": "12px", "marginRight": "16px"}),
                        html.Span(f"{doc['pages_analysed']} pages",
                                 style={"color": TEXT_DIM, "fontSize": "12px"}),
                    ]),
                ], style={
                    "background": CARD_BG, "border": f"1px solid {BORDER}",
                    "borderRadius": "8px", "padding": "16px",
                })
                for doc in DOC_ANALYSIS
            ], style={"display": "grid",
                      "gridTemplateColumns": "repeat(auto-fill, minmax(280px, 1fr))",
                      "gap": "12px", "marginTop": "16px"}),
        ])

    elif tab == "tab-signals":
        if SIGNALS_DF.empty:
            return html.P("No signals extracted.", style={"color": TEXT_DIM})
        return html.Div([
            dcc.Graph(figure=fig_severity_breakdown()),
            html.Div([
                html.Div([
                    html.Div([
                        html.Span(sig.get("severity",""), style={
                            "background": RISK_COLORS.get(sig.get("severity",""), BORDER),
                            "color": "white", "fontSize": "10px", "fontWeight": "700",
                            "padding": "2px 8px", "borderRadius": "4px",
                            "marginRight": "8px",
                        }),
                        html.Span(sig.get("risk_category","").replace("_"," ").upper(), style={
                            "color": CATEGORY_COLORS.get(sig.get("risk_category",""), TEXT_DIM),
                            "fontSize": "10px", "fontFamily": FONT_MONO,
                        }),
                    ], style={"marginBottom": "6px"}),
                    html.P(sig.get("description",""), style={
                        "color": TEXT, "fontSize": "13px", "margin": "0 0 6px",
                    }),
                    html.P(f'"{sig.get("evidence","")}"', style={
                        "color": TEXT_DIM, "fontSize": "11px",
                        "fontStyle": "italic", "margin": "0 0 4px",
                        "borderLeft": f"2px solid {BORDER}",
                        "paddingLeft": "8px",
                    }),
                    html.Div(sig.get("document",""), style={
                        "color": TEXT_DIM, "fontSize": "10px",
                        "fontFamily": FONT_MONO,
                    }),
                ], style={
                    "background": CARD_BG,
                    "border": f"1px solid {BORDER}",
                    "borderLeft": f"3px solid {RISK_COLORS.get(sig.get('severity',''), BORDER)}",
                    "borderRadius": "6px", "padding": "12px 14px",
                })
                for sig in ALL_SIGNALS
                if sig.get("severity") == "High"
            ], style={"display": "flex", "flexDirection": "column", "gap": "10px",
                      "marginTop": "16px"}),
        ])

    elif tab == "tab-validity":
        return html.Div([
            html.H3("Valimetrica Validity Assessment", style={
                "color": TEXT, "fontFamily": FONT_BODY, "marginBottom": "16px",
            }),
            html.P(
                "The following validity flags are raised for the quantitative scoring instruments "
                "applied to NZX/ASX companies. These reflect the Valimetrica framework's assessment "
                "of construct validity, operational validity, and decision validity.",
                style={"color": TEXT_DIM, "fontSize": "13px", "marginBottom": "20px"},
            ),
            html.Div([
                html.Div([
                    html.Div(title, style={
                        "fontFamily": FONT_MONO, "fontSize": "11px",
                        "color": ACCENT, "fontWeight": "600", "marginBottom": "8px",
                    }),
                    html.P(body, style={"color": TEXT, "fontSize": "13px", "margin": 0}),
                ], style={
                    "background": CARD_BG,
                    "border": f"1px solid {BORDER}",
                    "borderLeft": f"3px solid {WARN}",
                    "borderRadius": "6px", "padding": "14px 16px",
                })
                for title, body in [
                    ("BENEISH M-SCORE — CONSTRUCT VALIDITY",
                     "The Beneish M-score was calibrated on US Compustat data from the 1990s. "
                     "Its eight variables reflect US accounting standards (US GAAP) and market "
                     "structures. NZX and ASX companies report under NZ IFRS and AASB respectively. "
                     "Key differences in revenue recognition, lease accounting, and financial "
                     "instrument classification mean the score's discriminatory thresholds (-1.78, -2.22) "
                     "may not transfer. Treat M-score flags as indicative rather than definitive for "
                     "NZ/AU companies until local recalibration is performed."),
                    ("ALTMAN Z-SCORE — POPULATION VALIDITY",
                     "The original Altman Z-score (1968) was developed on 66 US manufacturing firms. "
                     "The NZX and ASX are dominated by financial services, real estate, and primary "
                     "industries — sectors structurally excluded from Altman's original sample. "
                     "Banks, REITs, and resource companies show systematically anomalous Z-scores "
                     "not because they are distressed but because the model's coefficients do not "
                     "apply to their balance sheet structures. This is a population validity failure: "
                     "the instrument is applied outside the population for which it was validated."),
                    ("PIOTROSKI F-SCORE — MOST ROBUST",
                     "The Piotroski F-score is the most transferable of the three instruments. "
                     "Its nine binary signals are based on fundamental accounting relationships "
                     "(profitability, leverage, efficiency) that are relatively stable across "
                     "accounting regimes. Coverage is highest (8/9 signals computable for most "
                     "NZX/ASX companies) and the directional interpretation is robust. "
                     "This is the primary signal recommended for NZ/AU regulatory screening."),
                    ("COMPOSITE RISK TIER — DECISION VALIDITY",
                     "The composite risk tier aggregates three scores with different validity "
                     "properties and different calibration populations. A 'High' tier flag driven "
                     "primarily by Beneish M should be weighted lower than one driven by both "
                     "Altman Z and Piotroski F. Regulatory decisions based on this composite "
                     "should treat it as a screening tool for prioritising further investigation, "
                     "not as a definitive risk determination. This distinction — between a screening "
                     "instrument and a decision instrument — is the core Valimetrica insight."),
                ]
            ], style={"display": "flex", "flexDirection": "column", "gap": "12px"}),
        ])


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8053))
    print(f"\n{'='*50}")
    print(f"  Financial Conduct Risk Analyser")
    print(f"  http://127.0.0.1:{port}")
    print(f"{'='*50}\n")
    app.run(debug=True, port=port, host="0.0.0.0")
