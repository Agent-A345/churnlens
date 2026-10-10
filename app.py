"""Streamlit dashboard: real-time churn prediction with SHAP explanations.

Run from the project root:
    streamlit run app.py
"""

import json

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import (
    CATEGORICAL_FEATURES,
    FEATURE_INFO_PATH,
    FEATURE_LABELS,
    FEATURES,
    FIGURES_DIR,
    METRICS_PATH,
    MODEL_PATH,
    NUMERIC_FEATURES,
    ROOT,
    TARGET,
)
from src.data_prep import fix_categories, load_clean
from src.explain import base_value, explain_customers, get_explainer, top_reasons

st.set_page_config(page_title="ChurnLens", page_icon="📉", layout="wide")

INK, SAFE, WATCH, RISK, MUTED, LINE = "#14213D", "#1B9E8F", "#E09A2D", "#C8344B", "#5B6578", "#DDE3EC"
FONT = dict(family="Manrope, system-ui, sans-serif", color="#1B2433")

css = (ROOT / "assets" / "style.css")
if css.exists():
    st.markdown(f"<style>{css.read_text()}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------- loading
@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists():
        st.error("No trained model found. Run `python -m src.train` first, then reload this page.")
        st.stop()
    pipe = joblib.load(MODEL_PATH)
    info = json.loads(FEATURE_INFO_PATH.read_text())
    metrics = json.loads(METRICS_PATH.read_text())
    return pipe, get_explainer(pipe), info, metrics


@st.cache_data
def load_data():
    try:
        return load_clean()
    except FileNotFoundError:
        return None


pipe, explainer, info, metrics = load_artifacts()
final = metrics["final_model"]
best = metrics["results"][final]


# ---------------------------------------------------------------- helpers
def risk_band(p: float):
    if p >= 0.7:
        return "At risk", RISK, "Very likely to leave. Reach out now with a retention offer or a support call."
    if p >= 0.4:
        return "Watch", WATCH, "Could go either way. Worth a check-in or a targeted coupon."
    return "Safe", SAFE, "Likely to stay. No action needed right now."


def style_fig(fig, height=320):
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=36, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=FONT, title_font=dict(size=15, color=INK),
    )
    fig.update_xaxes(gridcolor="#EEF1F6", zeroline=False)
    fig.update_yaxes(gridcolor="#EEF1F6", zeroline=False)
    return fig


def gauge(p: float):
    _, color, _ = risk_band(p)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=p * 100,
        number={"suffix": "%", "font": {"size": 46, "color": color}, "valueformat": ".1f"},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 0, "tickcolor": "white",
                     "tickvals": [0, 40, 70, 100], "tickfont": {"color": MUTED}},
            "bar": {"color": color, "thickness": 0.32},
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 40], "color": "#E3F3F1"},
                {"range": [40, 70], "color": "#FBF0DE"},
                {"range": [70, 100], "color": "#F7E1E5"},
            ],
        },
    ))
    return style_fig(fig, 250).update_layout(margin=dict(l=40, r=40, t=24, b=0))


def shap_bars(contrib: pd.Series, row: pd.Series, k: int = 10):
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(k)[::-1]
    labels = []
    for f in top.index:
        v = row[f]
        if isinstance(v, float) and v.is_integer():
            v = int(v)
        if f == "Complain":
            v = "Yes" if v == 1 else "No"
        labels.append(f"{FEATURE_LABELS[f].split(' (')[0]} = {v}")
    fig = go.Figure(go.Bar(
        x=top.values, y=labels, orientation="h",
        marker_color=[RISK if v > 0 else SAFE for v in top.values],
        text=[f"{v:+.2f}" for v in top.values], textposition="outside",
        hovertemplate="%{y}<br>effect %{x:+.3f}<extra></extra>",
        cliponaxis=False,
    ))
    fig.add_vline(x=0, line_color=INK, line_width=1)
    lo, hi = min(0.0, float(top.min())), max(0.0, float(top.max()))
    pad = (hi - lo) * 0.18 + 0.05
    fig.update_xaxes(range=[lo - (pad if lo < 0 else 0.02), hi + pad])
    fig.update_layout(title="What moved this prediction",
                      xaxis_title="Effect on churn score (SHAP)", showlegend=False)
    return style_fig(fig, 420)


def reasons_html(to_churn, to_stay):
    def items(rows, cls):
        if not rows:
            return '<div class="reason"><span>Nothing significant</span></div>'
        return "".join(f'<div class="reason {cls}"><span class="dot"></span><span>{t}</span></div>'
                       for t, _ in rows)
    return (f'<div class="reasons"><h4>Raising the risk</h4>{items(to_churn, "up")}'
            f'<h4>Keeping them</h4>{items(to_stay, "down")}</div>')


# ---------------------------------------------------------------- example presets
EXAMPLES = {
    "at_risk": {
        "Tenure": 1, "Complain": 1, "SatisfactionScore": 4, "DaySinceLastOrder": 1,
        "HourSpendOnApp": 3, "NumberOfDeviceRegistered": 5, "OrderCount": 2, "CouponUsed": 1,
        "CashbackAmount": 125.0, "OrderAmountHikeFromlastYear": 13, "CityTier": 3,
        "WarehouseToHome": 28, "NumberOfAddress": 8, "PreferedOrderCat": "Mobile Phone",
        "PreferredPaymentMode": "Cash on Delivery", "PreferredLoginDevice": "Computer",
        "Gender": "Male", "MaritalStatus": "Single",
    },
    "loyal": {
        "Tenure": 24, "Complain": 0, "SatisfactionScore": 3, "DaySinceLastOrder": 8,
        "HourSpendOnApp": 3, "NumberOfDeviceRegistered": 3, "OrderCount": 3, "CouponUsed": 2,
        "CashbackAmount": 230.0, "OrderAmountHikeFromlastYear": 16, "CityTier": 1,
        "WarehouseToHome": 9, "NumberOfAddress": 2, "PreferedOrderCat": "Laptop & Accessory",
        "PreferredPaymentMode": "Debit Card", "PreferredLoginDevice": "Mobile Phone",
        "Gender": "Female", "MaritalStatus": "Married",
    },
}


def default_values():
    vals = {c: info["numeric"][c]["median"] for c in NUMERIC_FEATURES}
    vals.update({c: info["categorical"][c]["mode"] for c in CATEGORICAL_FEATURES})
    return vals


def apply_values(vals):
    for col, v in vals.items():
        if col == "Complain":
            st.session_state[f"in_{col}"] = "Yes" if v == 1 else "No"
        elif col in NUMERIC_FEATURES:
            spec = info["numeric"][col]
            st.session_state[f"in_{col}"] = int(round(v)) if spec["is_int"] else float(v)
        else:
            st.session_state[f"in_{col}"] = v
    st.session_state.pop("result", None)


if "in_Tenure" not in st.session_state:
    apply_values(default_values())


def number_field(col):
    spec = info["numeric"][col]
    if col == "Complain":
        st.selectbox("Complained last month", ["No", "Yes"], key=f"in_{col}")
    elif spec["is_int"]:
        st.number_input(FEATURE_LABELS[col], int(spec["min"]), int(spec["max"]),
                        step=1, key=f"in_{col}")
    else:
        st.number_input(FEATURE_LABELS[col], float(spec["min"]), float(spec["max"]),
                        step=5.0, key=f"in_{col}")


def cat_field(col):
    st.selectbox(FEATURE_LABELS[col], info["categorical"][col]["options"], key=f"in_{col}")


def field(col):
    (cat_field if col in CATEGORICAL_FEATURES else number_field)(col)


GROUPS = [
    ("Relationship", "How long they've been around and how it's going",
     ["Tenure", "Complain", "SatisfactionScore", "DaySinceLastOrder", "HourSpendOnApp",
      "NumberOfDeviceRegistered"]),
    ("Orders and rewards", "What they buy and how they pay",
     ["OrderCount", "CouponUsed", "CashbackAmount", "OrderAmountHikeFromlastYear",
      "PreferedOrderCat", "PreferredPaymentMode"]),
    ("Profile and location", "Who they are and where they live",
     ["CityTier", "WarehouseToHome", "NumberOfAddress", "PreferredLoginDevice",
      "Gender", "MaritalStatus"]),
]


# ---------------------------------------------------------------- header
st.markdown(f"""
<div class="hero">
  <div>
    <h1>ChurnLens</h1>
    <p>Find e-commerce customers who are about to leave, and see exactly why,
       using XGBoost with SHAP explanations.</p>
  </div>
  <div class="hero-stats">
    <div class="hero-stat"><div class="v">{best['f1']:.3f}</div><div class="l">F1-score</div></div>
    <div class="hero-stat"><div class="v">{best['roc_auc']:.3f}</div><div class="l">ROC-AUC</div></div>
    <div class="hero-stat"><div class="v">{best['recall']:.0%}</div><div class="l">Churners caught</div></div>
    <div class="hero-stat"><div class="v">{metrics['n_test']:,}</div><div class="l">Test customers</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

tab_one, tab_many, tab_insights, tab_model = st.tabs(
    ["Score a customer", "Score a list", "Customer insights", "Model report"])

# ---------------------------------------------------------------- score one
with tab_one:
    b1, b2, b3, _ = st.columns([1.3, 1.2, 1, 3])
    if b1.button("Load at-risk example", use_container_width=True):
        apply_values(EXAMPLES["at_risk"])
        st.rerun()
    if b2.button("Load loyal example", use_container_width=True):
        apply_values(EXAMPLES["loyal"])
        st.rerun()
    if b3.button("Reset", use_container_width=True):
        apply_values(default_values())
        st.rerun()

    with st.form("customer"):
        cols = st.columns(3, gap="medium")
        for col_box, (title, sub, feats) in zip(cols, GROUPS):
            with col_box:
                st.markdown(f'<p class="panel-title">{title}</p><p class="panel-sub">{sub}</p>',
                            unsafe_allow_html=True)
                for f in feats:
                    field(f)
        submitted = st.form_submit_button("Predict churn", type="primary")

    if submitted:
        vals = {}
        for f in FEATURES:
            v = st.session_state[f"in_{f}"]
            vals[f] = (1 if v == "Yes" else 0) if f == "Complain" else v
        X = pd.DataFrame([vals])[FEATURES]
        p = float(pipe.predict_proba(X)[0, 1])
        contrib = explain_customers(pipe, explainer, X).iloc[0]
        st.session_state["result"] = (X, p, contrib)

    if "result" in st.session_state:
        X, p, contrib = st.session_state["result"]
        label, color, advice = risk_band(p)
        to_churn, to_stay = top_reasons(contrib, X.iloc[0])

        left, right = st.columns([1, 1.45], gap="large")
        with left:
            with st.container(border=True):
                st.markdown('<p class="panel-title">Churn probability</p>', unsafe_allow_html=True)
                st.plotly_chart(gauge(p), use_container_width=True,
                                config={"displayModeBar": False})
                st.markdown(f'<div class="verdict" style="--c:{color}"><p class="t">{label}</p>'
                            f'<p class="d">{advice}</p></div>', unsafe_allow_html=True)
                st.markdown(reasons_html(to_churn, to_stay), unsafe_allow_html=True)
        with right:
            with st.container(border=True):
                st.plotly_chart(shap_bars(contrib, X.iloc[0]), use_container_width=True,
                                config={"displayModeBar": False})
                st.markdown(
                    f'<p class="note">Red bars push this customer toward leaving, teal bars '
                    f'hold them back. Values are SHAP contributions in log-odds, measured '
                    f'from the average customer ({base_value(explainer):+.2f}).</p>',
                    unsafe_allow_html=True)

# ---------------------------------------------------------------- score many
with tab_many:
    st.markdown('<p class="panel-title">Score a whole customer list</p>'
                '<p class="panel-sub">Upload a CSV or Excel file with the same columns as the '
                'training data. Every customer gets a churn probability, a risk band and their '
                'main reason.</p>', unsafe_allow_html=True)
    file = st.file_uploader("Customer file", type=["csv", "xlsx"], label_visibility="collapsed")
    if file is None:
        st.info("Tip: upload `data/E Commerce Dataset.xlsx` to try it on the full dataset.")
    else:
        df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file, sheet_name=-1)
        missing = [c for c in FEATURES if c not in df.columns]
        if missing:
            st.error(f"This file is missing {len(missing)} required column(s): "
                     f"{', '.join(missing)}. Add them and upload again.")
        else:
            X = fix_categories(df)[FEATURES]
            proba = pipe.predict_proba(X)[:, 1]
            contrib = explain_customers(pipe, explainer, X)
            out = df.drop(columns=[TARGET], errors="ignore").copy()
            out.insert(0, "Main reason", [
                FEATURE_LABELS[r.idxmax()].split(" (")[0] if r.max() > 0 else "None"
                for _, r in contrib.iterrows()])
            out.insert(0, "Risk", [risk_band(p)[0] for p in proba])
            out.insert(0, "Churn probability", proba.round(4))
            out = out.sort_values("Churn probability", ascending=False)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Customers scored", f"{len(out):,}")
            c2.metric("At risk", f"{(out['Risk'] == 'At risk').sum():,}")
            c3.metric("Watch", f"{(out['Risk'] == 'Watch').sum():,}")
            c4.metric("Average probability", f"{proba.mean():.1%}")

            g1, g2 = st.columns([1, 1.4], gap="large")
            with g1:
                counts = out["Risk"].value_counts().reindex(["Safe", "Watch", "At risk"]).fillna(0)
                fig = go.Figure(go.Pie(labels=counts.index, values=counts.values, hole=0.62,
                                       marker_colors=[SAFE, WATCH, RISK], sort=False,
                                       texttemplate="%{label}<br>%{percent:.1%}",
                                       textposition="outside"))
                fig.update_layout(title="Risk mix", showlegend=False)
                st.plotly_chart(style_fig(fig, 300), use_container_width=True,
                                config={"displayModeBar": False})
            with g2:
                reasons = out.loc[out["Risk"] != "Safe", "Main reason"].value_counts().head(6)[::-1]
                fig = go.Figure(go.Bar(x=reasons.values, y=reasons.index, orientation="h",
                                       marker_color=INK, text=reasons.values,
                                       textposition="outside", cliponaxis=False))
                fig.update_layout(title="Main reason among at-risk and watch customers",
                                  xaxis_title="Customers")
                st.plotly_chart(style_fig(fig, 300), use_container_width=True,
                                config={"displayModeBar": False})

            st.dataframe(
                out, use_container_width=True, height=420, hide_index=True,
                column_config={"Churn probability": st.column_config.ProgressColumn(
                    "Churn probability", format="%.2f", min_value=0, max_value=1)})
            st.download_button("Download scored list (CSV)", out.to_csv(index=False).encode(),
                               "churn_predictions.csv", "text/csv")

# ---------------------------------------------------------------- insights
with tab_insights:
    data = load_data()
    if data is None:
        st.warning("Put `E Commerce Dataset.xlsx` in the data folder to see the insights.")
    else:
        rate = data[TARGET].mean()
        new = data["Tenure"] <= 1
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Customers (after cleaning)", f"{len(data):,}")
        c2.metric("Overall churn rate", f"{rate:.1%}")
        c3.metric("Churn in first 2 months", f"{data.loc[new, TARGET].mean():.0%}")
        c4.metric("Churn after complaint",
                  f"{data.loc[data['Complain'] == 1, TARGET].mean():.0%}",
                  f"{data.loc[data['Complain'] == 1, TARGET].mean() - data.loc[data['Complain'] == 0, TARGET].mean():+.0%} vs no complaint",
                  delta_color="inverse")

        def rate_bar(series, title, order=None):
            r = data.groupby(series, observed=True)[TARGET].mean()
            if order is not None:
                r = r.reindex(order)
            else:
                r = r.sort_values(ascending=False)
            colors = [RISK if v > rate else "#9AA7BD" for v in r.values]
            fig = go.Figure(go.Bar(x=[str(i) for i in r.index], y=r.values, marker_color=colors,
                                   text=[f"{v:.0%}" for v in r.values], textposition="outside",
                                   cliponaxis=False))
            fig.add_hline(y=rate, line_dash="dot", line_color=INK,
                          annotation_text=f"average {rate:.0%}", annotation_position="top right",
                          annotation_font_color=MUTED)
            fig.update_yaxes(range=[0, max(r.max(), rate) * 1.18])
            fig.update_layout(title=title, yaxis_tickformat=".0%", showlegend=False)
            return style_fig(fig, 330)

        tenure_group = pd.cut(data["Tenure"], [-1, 0, 1, 3, 6, 12, 24, 100],
                              labels=["0", "1", "2-3", "4-6", "7-12", "13-24", "25+"])
        tenure_group.name = "Tenure (months)"

        st.markdown('<p class="note">Red bars are above the average churn rate. '
                    'Hover any bar for the exact value.</p>', unsafe_allow_html=True)
        r1a, r1b = st.columns([1.4, 1], gap="large")
        r1a.plotly_chart(rate_bar(tenure_group, "Churn rate by months with the company",
                                  order=list(tenure_group.cat.categories)),
                         use_container_width=True, config={"displayModeBar": False})
        r1b.plotly_chart(rate_bar(data["Complain"].map({0: "No complaint", 1: "Complained"}),
                                  "Churn rate by complaint"),
                         use_container_width=True, config={"displayModeBar": False})
        r2a, r2b = st.columns(2, gap="large")
        r2a.plotly_chart(rate_bar(data["PreferedOrderCat"], "Churn rate by order category"),
                         use_container_width=True, config={"displayModeBar": False})
        r2b.plotly_chart(rate_bar(data["PreferredPaymentMode"], "Churn rate by payment mode"),
                         use_container_width=True, config={"displayModeBar": False})
        r3a, r3b = st.columns(2, gap="large")
        r3a.plotly_chart(rate_bar(data["MaritalStatus"], "Churn rate by marital status"),
                         use_container_width=True, config={"displayModeBar": False})
        r3b.plotly_chart(rate_bar(data["SatisfactionScore"], "Churn rate by satisfaction score",
                                  order=[1, 2, 3, 4, 5]),
                         use_container_width=True, config={"displayModeBar": False})
        st.markdown('<p class="note">Note: churn rises with satisfaction score in this dataset. '
                    'This is counter-intuitive and is reported as a limitation.</p>',
                    unsafe_allow_html=True)

# ---------------------------------------------------------------- model report
with tab_model:
    table = pd.DataFrame(metrics["results"]).T[["accuracy", "precision", "recall", "f1", "roc_auc"]]
    table.columns = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]

    m1, m2 = st.columns([1.5, 1], gap="large")
    with m1:
        fig = go.Figure()
        palette = ["#B8C2D3", "#7D8DAA", "#4A5F86", INK]
        for name, color in zip(table.index, palette):
            fig.add_bar(name=name, x=table.columns, y=table.loc[name], marker_color=color,
                        hovertemplate=f"{name}<br>%{{x}}: %{{y:.4f}}<extra></extra>")
        fig.update_layout(title="Model comparison on 1,015 unseen customers", barmode="group",
                          yaxis_range=[0, 1.05], legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(style_fig(fig, 380), use_container_width=True,
                        config={"displayModeBar": False})
    with m2:
        cm = metrics["confusion_matrix"]
        z = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
        fig = go.Figure(go.Heatmap(
            z=z, x=["Predicted stay", "Predicted churn"], y=["Actually stayed", "Actually churned"],
            colorscale=[[0, "#EEF1F6"], [1, INK]], showscale=False,
            text=[[f"{v:,}" for v in r] for r in z], texttemplate="%{text}",
            textfont={"size": 20}))
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(title=f"Confusion matrix: {final}")
        st.plotly_chart(style_fig(fig, 380), use_container_width=True,
                        config={"displayModeBar": False})

    st.dataframe(table.style.format("{:.4f}").highlight_max(axis=0, color="#DCEFEC"),
                 use_container_width=True)
    st.markdown(f'<p class="note">The final model, {final}, was chosen on cross-validation F1, '
                'so the test set was only used once, for these numbers.</p>',
                unsafe_allow_html=True)

    figs = [("shap_summary.png", "What drives churn across all customers (SHAP)"),
            ("roc_curves.png", "ROC curves")]
    cols = st.columns(2, gap="large")
    for col, (fname, title) in zip(cols, figs):
        path = FIGURES_DIR / fname
        if path.exists():
            with col:
                with st.container(border=True):
                    st.markdown(f'<p class="panel-title">{title}</p>', unsafe_allow_html=True)
                    st.image(str(path), use_container_width=True)
