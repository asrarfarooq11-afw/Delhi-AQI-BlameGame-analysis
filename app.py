import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Delhi Air Pollution | Asrar",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATASET_URL = "https://www.kaggle.com/datasets/abhisheksjha/time-series-air-quality-data-of-india-2010-2023"

# ----------------------------- Theme -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif;
}
.hero {
    padding: 2.2rem 2.4rem;
    border-radius: 24px;
    background: linear-gradient(135deg, #eef2ff 0%, #f8fafc 55%, #ecfeff 100%);
    border: 1px solid #e2e8f0;
    margin-bottom: 1.2rem;
}
.hero h1 { font-size: 3rem; margin-bottom: .35rem; }
.hero p { color: #475569; font-size: 1.08rem; max-width: 900px; }
.kicker {
    color: #6366f1; font-weight: 700; letter-spacing: .08em;
    text-transform: uppercase; font-size: .78rem;
}
.card {
    padding: 1.1rem 1.2rem;
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    background: white;
    height: 100%;
}
.big { font-size: 2rem; font-weight: 700; font-family: 'Space Grotesk'; }
.muted { color: #64748b; }
.pill {
    display:inline-block; padding:.28rem .65rem; border-radius:999px;
    background:#f1f5f9; color:#334155; font-size:.78rem; font-weight:600;
}
.blame {
    padding: 1.1rem; border-radius: 18px; background:#0f172a; color:white;
    min-height: 145px;
}
.blame b { color:#a5b4fc; }
.insight {
    padding: 1rem 1.15rem; border-left: 4px solid #6366f1;
    background:#f8fafc; border-radius: 0 14px 14px 0; margin:.45rem 0;
}
.small-note { color:#64748b; font-size:.85rem; }
div[data-testid="stMetric"] {
    background:#fff; border:1px solid #e2e8f0; padding:1rem;
    border-radius:16px;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------- Data -----------------------------
corr = pd.DataFrame({
    "Feature": [
        "Punjab · 2-day lag", "Punjab · 1-day lag",
        "Haryana · 1-day lag", "Haryana · 2-day lag"
    ],
    "Era 1": [0.553, 0.591, 0.658, 0.558],
    "Era 2": [0.716, 0.779, 0.855, 0.747],
})

covid = pd.DataFrame({
    "Phase": ["Pre-COVID", "Lockdown", "Post-COVID"],
    "Delhi": [109.4, 50.4, 100.0],
    "Punjab": [57.1, 32.4, 52.4],
    "Haryana": [91.0, 40.9, 72.6],
})

models = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest", "XGBoost"],
    "RMSE": [34.61, 29.08, 29.74],
    "R²": [0.736, 0.814, 0.805],
})

ablation = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest", "XGBoost"],
    "Full": [0.736494, 0.813889, 0.805449],
    "No Punjab": [0.651156, 0.789062, 0.777643],
    "No Haryana": [0.656, 0.655, 0.668],
})
ablation["Punjab drop %"] = (ablation["Full"] - ablation["No Punjab"]) / ablation["Full"] * 100
ablation["Haryana drop %"] = (ablation["Full"] - ablation["No Haryana"]) / ablation["Full"] * 100

importance = pd.DataFrame({
    "Feature": [
        "Haryana · 1-day lag", "month", "Punjab · 3-day average",
        "Punjab · 1-day lag", "Punjab · 2-day lag",
        "Haryana · 2-day lag", "stubble season", "day of week", "weekend"
    ],
    "Importance": [0.613, 0.146, 0.083, 0.049, 0.041, 0.039, 0.014, 0.013, 0.001]
})

classification = pd.DataFrame({
    "Model": ["Logistic Regression", "SVM"],
    "Accuracy": [0.899, 0.8797978979],
    "Precision": [0.675, 0.604],
    "Recall": [0.878, 0.943],
    "F1": [0.763, None],
})

# ----------------------------- Sidebar -----------------------------
st.sidebar.markdown("## 🌫️ Air Pollution Lab")
st.sidebar.caption("A portfolio dashboard by **Asrar**")
page = st.sidebar.radio(
    "Explore",
    ["Story", "Data & Method", "Correlation", "ML", "Severe Days", "Takeaways"],
)
st.sidebar.divider()
st.sidebar.markdown("**Dataset**")
st.sidebar.link_button("Open Kaggle dataset ↗", DATASET_URL)
st.sidebar.caption("Time Series Air Quality Data of India, 2010–2023")

# ----------------------------- Hero -----------------------------
st.markdown("""
<div class="hero">
  <div class="kicker">Data × Statistics × Machine Learning</div>
  <h1>Delhi's Air Pollution: Who's Actually to Blame?</h1>
  <p>
    A station-aware investigation into whether Punjab and Haryana pollution
    contains time-lagged signals associated with Delhi PM2.5 — and whether
    those signals can help predict pollution and severe days.
  </p>
  <span class="pill">Project by Asrar</span>
  <span class="pill">70/30 temporal split</span>
  <span class="pill">2017–2023 common period</span>
</div>
""", unsafe_allow_html=True)

# ----------------------------- Story -----------------------------
if page == "Story":
    st.subheader("🕷️ The 'blame game'")

    a, b, c = st.columns(3)
    with a:
        st.markdown("""
        <div class="blame">
        <b>Punjab:</b><br><br>
        "Is it really all on us?"
        </div>
        """, unsafe_allow_html=True)
    with b:
        st.markdown("""
        <div class="blame">
        <b>Haryana:</b><br><br>
        "Bro, check the lag features."
        </div>
        """, unsafe_allow_html=True)
    with c:
        st.markdown("""
        <div class="blame">
        <b>Delhi:</b><br><br>
        "Okay… let's ask the data. 💀"
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### The investigation")
    cols = st.columns(5)
    steps = [
        ("01", "Stations", "Validate when stations were operational"),
        ("02", "Daily data", "Average hourly PM2.5 by day"),
        ("03", "Lag signals", "Punjab −2d, Haryana −1d"),
        ("04", "Statistics", "Correlation, eras, COVID, stubble"),
        ("05", "ML", "Regression + severe-day classification"),
    ]
    for col, (n, title, text) in zip(cols, steps):
        with col:
            st.markdown(f'<div class="card"><span class="pill">{n}</span><h4>{title}</h4><span class="muted">{text}</span></div>', unsafe_allow_html=True)

    st.markdown("### The headline")
    x, y, z = st.columns(3)
    with x:
        st.metric("Punjab lag-2 correlation", "0.716", "+29.5% vs Era 1")
    with y:
        st.metric("Best regression R²", "0.814", "Random Forest")
    with z:
        st.metric("SVM severe-day recall", "94.3%", "high recall")

    st.markdown("""
    <div class="insight">
    <b>Plot twist:</b> Punjab matters, but the model does not point to Punjab alone.
    Haryana's 1-day lag is the dominant Random Forest feature in this analysis.
    </div>
    """, unsafe_allow_html=True)

# ----------------------------- Data & Method -----------------------------
elif page == "Data & Method":
    st.subheader("🧪 From messy station files to one analysis table")

    c1, c2, c3, c4 = st.columns(4)
    for col, value, label in [
        (c1, "453", "station metadata rows"),
        (c2, "19", "Delhi stations used"),
        (c3, "20", "Haryana stations used"),
        (c4, "8", "Punjab stations used"),
    ]:
        with col:
            st.metric(label, value)

    st.markdown("### Pipeline")
    st.code(
        "Raw station CSVs\n"
        "   ↓\n"
        "Station-aware filtering (start_year)\n"
        "   ↓\n"
        "Daily PM2.5 averages\n"
        "   ↓\n"
        "Common period: 2017-03-03 onward\n"
        "   ↓\n"
        "Lags + rolling averages + calendar features\n"
        "   ↓\n"
        "Correlation + ML",
        language="text",
    )

    st.markdown("### Feature logic")
    f1, f2 = st.columns(2)
    with f1:
        st.markdown("""
        **Primary temporal signals**
        - Punjab PM2.5, **2 days earlier**
        - Haryana PM2.5, **1 day earlier**
        - Extra 1-day/2-day lags as robustness checks
        """)
    with f2:
        st.markdown("""
        **Context**
        - Punjab 3-day average
        - Month + day of week + weekend
        - Oct 15–Nov 30 stubble-season flag
        - Era and COVID phase
        """)

    st.info(
        "The lags create temporal predictive features. They support temporal association; "
        "they do not by themselves prove that one state caused Delhi's pollution."
    )

# ----------------------------- Correlation -----------------------------
elif page == "Correlation":
    st.subheader("🔗 Correlation: does the relationship change over time?")

    left, right = st.columns([2, 1])
    with left:
        plot = corr.melt(id_vars="Feature", var_name="Era", value_name="Pearson r")
        fig = px.bar(
            plot, x="Feature", y="Pearson r", color="Era", barmode="group",
            text_auto=".3f", template="simple_white",
        )
        fig.update_layout(height=480, yaxis_range=[0, 1], xaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("""
        <div class="card">
        <h3>Key finding 👀</h3>
        <div class="big">0.553 → 0.716</div>
        <p class="muted">Punjab 2-day lag, Era 1 → Era 2</p>
        <hr>
        <b>+29.5%</b> stronger Pearson correlation.
        <p class="small-note">
        The association strengthened in the later period.
        </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Stubble-season lens")
    stubble = pd.DataFrame({
        "Period": ["Era 1 · Stubble", "Era 1 · Non-stubble", "Era 2 · Stubble", "Era 2 · Non-stubble"],
        "r": [0.437, 0.444, 0.465, 0.692],
    })
    fig2 = px.bar(stubble, x="Period", y="r", text_auto=".3f", template="simple_white")
    fig2.update_layout(height=360, yaxis_range=[0, 0.8])
    st.plotly_chart(fig2, use_container_width=True)

    st.caption(
        "Interesting result: in Era 2, the Punjab–Delhi correlation was stronger outside "
        "the defined stubble-season window (r = 0.692) than inside it (r = 0.465)."
    )

# ----------------------------- ML -----------------------------
elif page == "ML":
    st.subheader("🤖 ML has entered the chat")

    fig = px.bar(
        models, x="Model", y="R²", text_auto=".3f",
        template="simple_white", title="Regression performance on the test set"
    )
    fig.update_layout(yaxis_range=[0.65, 0.85], height=400)
    st.plotly_chart(fig, use_container_width=True)

    m1, m2, m3 = st.columns(3)
    for col, row in zip([m1, m2, m3], models.itertuples()):
        with col:
            st.markdown(f"""
            <div class="card">
            <h4>{row.Model}</h4>
            <div class="big">{row._2:.3f}</div>
            <span class="muted">Test R²</span><br>
            RMSE: <b>{row.RMSE:.2f}</b>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("### What did the model actually learn?")
    fig2 = px.bar(
        importance.sort_values("Importance"),
        x="Importance", y="Feature", orientation="h",
        text_auto=".3f", template="simple_white"
    )
    fig2.update_layout(height=500, xaxis_range=[0, 0.67])
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("""
    <div class="insight">
    <b>Big reveal:</b> Haryana's 1-day lag is the dominant Random Forest feature
    (≈61.3% importance). Punjab is useful, but it is not the whole story.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Ablation = remove a group and see what breaks")
    fig3 = go.Figure()
    fig3.add_bar(name="Full", x=ablation.Model, y=ablation.Full)
    fig3.add_bar(name="Without Punjab", x=ablation.Model, y=ablation["No Punjab"])
    fig3.add_bar(name="Without Haryana", x=ablation.Model, y=ablation["No Haryana"])
    fig3.update_layout(
        barmode="group", template="simple_white", height=430,
        yaxis_title="Test R²", yaxis_range=[0.5, 0.85]
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.dataframe(
        ablation[["Model", "Punjab drop %", "Haryana drop %"]].round(2),
        hide_index=True, use_container_width=True
    )

    st.caption(
        "Ablation percentages are reductions in model R² relative to the full model. "
        "They are not percentages of Delhi pollution caused by a state."
    )

# ----------------------------- Severe Days -----------------------------
elif page == "Severe Days":
    st.subheader("🚨 Can we spot severe pollution days?")

    a, b, c = st.columns(3)
    with a:
        st.metric("Severe-day rule", "PM2.5 > 150")
    with b:
        st.metric("Severe days", "451 / 2,218", "20.3%")
    with c:
        st.metric("SVM accuracy", "88.0%")

    st.markdown("### Logistic Regression vs SVM")
    display_df = classification.copy()
    display_df["Accuracy"] = display_df["Accuracy"].map(lambda x: f"{x*100:.1f}%")
    display_df["Precision"] = display_df["Precision"].map(lambda x: f"{x*100:.1f}%")
    display_df["Recall"] = display_df["Recall"].map(lambda x: f"{x*100:.1f}%")
    display_df["F1"] = display_df["F1"].map(lambda x: "—" if pd.isna(x) else f"{x*100:.1f}%")
    st.dataframe(display_df, hide_index=True, use_container_width=True)

    fig = px.bar(
        classification.melt(id_vars="Model", value_vars=["Accuracy", "Precision", "Recall"],
                            var_name="Metric", value_name="Score"),
        x="Metric", y="Score", color="Model", barmode="group",
        text_auto=".1%", template="simple_white"
    )
    fig.update_layout(yaxis_range=[0.5, 1], height=430)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="insight">
    <b>The trade-off:</b> Logistic Regression has slightly higher accuracy and precision.
    SVM has <b>94.3% recall</b>, so it catches more of the truly severe days.
    For an early-warning use case, high recall can be more valuable than a small gain in accuracy.
    </div>
    """, unsafe_allow_html=True)

    st.warning(
        "The current notebook predicts the same-date severe label using lagged features. "
        "For a formally deployed 2-day-ahead target, the Delhi target should be shifted "
        "forward by two days during dataset construction."
    )

# ----------------------------- Takeaways -----------------------------
elif page == "Takeaways":
    st.subheader("🧠 If you remember only 6 things…")

    takeaways = [
        ("01", "Punjab–Delhi association strengthened", "Punjab 2-day lag: r = 0.553 → 0.716 (+29.5%)."),
        ("02", "Haryana is a major signal", "Haryana 1-day lag: r = 0.855 in Era 2 and ~61.3% RF feature importance."),
        ("03", "Random Forest wins regression", "Test R² ≈ 0.814 with RMSE ≈ 29.08."),
        ("04", "Punjab still adds predictive value", "Removing Punjab reduced RF R² from ~0.814 to ~0.789 (~3.1% relative reduction)."),
        ("05", "Haryana removal hurts more", "Removing Haryana reduced RF R² to ~0.655 (~19.7% relative reduction)."),
        ("06", "SVM catches severe days", "Accuracy ≈ 88.0%, recall = 94.3%."),
    ]
    for n, title, text in takeaways:
        st.markdown(f"""
        <div class="insight">
        <span class="pill">{n}</span> <b>{title}</b><br>
        <span class="muted">{text}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Final answer to the blame game")
    st.markdown("""
    <div class="card">
    <h3>It's not a one-state story.</h3>
    <p>
    The analysis finds meaningful time-lagged associations between neighboring-state
    pollution and Delhi PM2.5. Punjab matters, but Haryana shows stronger predictive
    value in this feature set. Local and regional factors both remain relevant.
    </p>
    <p class="small-note">
    Statistical association + predictive modelling ≠ causal proof.
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### COVID lens")
    fig = px.bar(
        covid.melt(id_vars="Phase", var_name="State", value_name="PM2.5"),
        x="Phase", y="PM2.5", color="State", barmode="group",
        text_auto=".1f", template="simple_white"
    )
    fig.update_layout(height=430, yaxis_title="Average PM2.5 (µg/m³)")
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Observed Delhi average: 109.4 pre-COVID → 50.4 during lockdown → 100.0 post-COVID "
        "(about −53.9% during lockdown vs pre-COVID)."
    )

st.divider()
st.caption(
    "Built by Asrar • Analysis uses station-aware filtering, daily aggregation, lag features, "
    "dual-era correlation, ablation studies, regression, and classification."
)
