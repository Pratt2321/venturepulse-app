import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.model_loader import load_data, load_model_bundle, predict_valuation, find_closest_comps

# --- Page Config ---
st.set_page_config(
    page_title="VenturePulse | Global Startup Valuation Intelligence",
    page_icon="🦄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Dark Aesthetic CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E1B4B 50%, #0F172A 100%);
        padding: 24px 32px;
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
    }
    
    .badge-pill {
        display: inline-block;
        padding: 4px 12px;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        border-radius: 20px;
        background: rgba(0, 242, 254, 0.15);
        color: #00F2FE;
        border: 1px solid rgba(0, 242, 254, 0.4);
        margin-bottom: 10px;
    }
    
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #FFFFFF, #CBD5E1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        padding-bottom: 6px;
    }
    
    .hero-subtitle {
        color: #94A3B8;
        font-size: 1.0rem;
        margin: 0;
    }

    .centicorn-spotlight {
        background: linear-gradient(90deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(0, 242, 254, 0.25);
        border-radius: 10px;
        padding: 12px 18px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 16px;
        flex-wrap: wrap;
    }

    .spotlight-badge {
        font-weight: 800;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        color: #00F2FE;
        background: rgba(0, 242, 254, 0.1);
        padding: 3px 10px;
        border-radius: 6px;
        border: 1px solid rgba(0, 242, 254, 0.3);
    }

    .spotlight-item {
        font-size: 0.88rem;
        color: #E2E8F0;
    }
    
    .stat-card {
        background: #1E293B;
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 18px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    .stat-card:hover {
        border-color: rgba(0, 242, 254, 0.5);
        transform: translateY(-2px);
    }
    
    .stat-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        color: #94A3B8;
        font-weight: 600;
        margin-bottom: 4px;
    }
    
    .stat-value {
        font-size: 1.6rem;
        font-weight: 800;
        color: #F8FAFC;
    }
    
    .stat-subtext {
        font-size: 0.75rem;
        color: #38BDF8;
        margin-top: 4px;
    }
    
    .comp-card {
        background: #0F172A;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 14px;
        margin-bottom: 12px;
    }
    
    .comp-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #38BDF8;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 8px;
        background-color: #1E293B;
        color: #94A3B8;
        font-weight: 600;
        border: 1px solid transparent;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #0284C7 !important;
        color: #FFFFFF !important;
        border: 1px solid #38BDF8 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Load Data & Model Bundle ---
try:
    df = load_data()
    bundle = load_model_bundle()
except Exception as e:
    st.error(f"Error loading system files: {e}. Please ensure data/download_and_clean.py and models/train_model.py have executed.")
    st.stop()

# --- Top Banner Header ---
st.markdown("""
<div class="main-header">
    <div class="badge-pill">Venture Capital & Private Equity Intelligence</div>
    <h1 class="hero-title">🦄 VenturePulse: Startup Valuation & Market Intelligence</h1>
    <p class="hero-subtitle">
        Reverse-engineering private tech valuations across 1,447 global unicorns (featuring Anthropic, OpenAI, ByteDance, and Databricks) using empirical ML regression and investor network elasticity.
    </p>
</div>
""", unsafe_allow_html=True)

# --- Centicorn Spotlight Row ---
st.markdown("""
<div class="centicorn-spotlight">
    <span class="spotlight-badge">🔥 Mega-Centicorn Vanguard</span>
    <span class="spotlight-item">🤖 <strong>Anthropic</strong>: <span style="color:#00F2FE;">$965B</span> (SF)</span>
    <span style="color:#475569;">•</span>
    <span class="spotlight-item">🧠 <strong>OpenAI</strong>: <span style="color:#00F2FE;">$840B</span> (SF)</span>
    <span style="color:#475569;">•</span>
    <span class="spotlight-item">📱 <strong>ByteDance</strong>: <span style="color:#00F2FE;">$480B</span> (Beijing)</span>
    <span style="color:#475569;">•</span>
    <span class="spotlight-item">💾 <strong>Databricks</strong>: <span style="color:#00F2FE;">$190B</span> (SF)</span>
    <span style="color:#475569;">•</span>
    <span class="spotlight-item">💳 <strong>Stripe</strong>: <span style="color:#00F2FE;">$159B</span> (SF)</span>
    <span style="color:#475569;">•</span>
    <span class="spotlight-item">⚡ <strong>DeepSeek</strong>: <span style="color:#00F2FE;">$50B</span> (Hangzhou)</span>
</div>
""", unsafe_allow_html=True)

# --- High Level KPIs ---
k1, k2, k3, k4, k5 = st.columns(5)
total_market_cap = df["valuation_b"].sum()
with k1:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">Total Ecosystem Value</div>
        <div class="stat-value">${total_market_cap/1000.0:.2f} Trillion</div>
        <div class="stat-subtext">Across {len(df):,} global unicorns</div>
    </div>
    """, unsafe_allow_html=True)
with k2:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">Median Valuation</div>
        <div class="stat-value">${df['valuation_b'].median():.2f} Billion</div>
        <div class="stat-subtext">Baseline entry to unicorn club</div>
    </div>
    """, unsafe_allow_html=True)
with k3:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-label">Frontier Centicorns</div>
        <div class="stat-value">Anthropic & OpenAI</div>
        <div class="stat-subtext">$965B & $840B Mega-Leaders</div>
    </div>
    """, unsafe_allow_html=True)
with k4:
    bay_count = (df['hub_category'] == 'SF Bay Area').sum()
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">Leading Tech Hub</div>
        <div class="stat-value">SF Bay Area</div>
        <div class="stat-subtext">{bay_count:,} titans in Silicon Valley</div>
    </div>
    """, unsafe_allow_html=True)
with k5:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-label">Largest Sector</div>
        <div class="stat-value">Enterprise Tech</div>
        <div class="stat-subtext">561 active titans (AI & SaaS)</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# --- Sidebar Controls ---
st.sidebar.image("https://img.icons8.com/fluency/96/unicorn.png", width=64)
st.sidebar.title("VenturePulse Filters")
st.sidebar.markdown(f"Exploring **{len(df):,}** live tech unicorns worldwide.")

selected_industry = st.sidebar.selectbox(
    "Filter by Industry Sector:",
    ["All Sectors"] + sorted(df["clean_industry"].unique().tolist())
)

selected_hub = st.sidebar.selectbox(
    "Filter by Tech Hub:",
    ["All Hubs"] + sorted(df["hub_category"].unique().tolist())
)

only_ai = st.sidebar.checkbox("Show Only Frontier AI & DeepTech", value=False)

min_val = st.sidebar.slider(
    "Minimum Valuation ($B):",
    min_value=1.0,
    max_value=float(df["valuation_b"].max()),
    value=1.0,
    step=5.0
)

# Apply global filters for visualizations
filtered_df = df.copy()
if selected_industry != "All Sectors":
    filtered_df = filtered_df[filtered_df["clean_industry"] == selected_industry]
if selected_hub != "All Hubs":
    filtered_df = filtered_df[filtered_df["hub_category"] == selected_hub]
if only_ai:
    filtered_df = filtered_df[filtered_df["is_ai_frontier"] == 1]
filtered_df = filtered_df[filtered_df["valuation_b"] >= min_val]

# --- Main Tabs ---
tab1, tab2, tab3, tab4 = st.tabs([
    "🗺️ Global Unicorn Map",
    "🎛️ The VC Valuation Simulator",
    "⚡ Velocity & Decacorn Screener",
    "📈 Model Diagnostics & Elasticity"
])

# =====================================================================
# TAB 1: Global Map & Macro Landscape
# =====================================================================
with tab1:
    st.subheader("Global Headquarters of the World's Billion-Dollar Startups")
    st.markdown(f"Displaying **{len(filtered_df):,}** active unicorns. Circles are scaled with visual logarithmic expansion so mega-centicorns like **Anthropic ($965B)** and **OpenAI ($840B)** command prominent prominence.")

    map_c1, map_c2 = st.columns([1, 4])
    with map_c1:
        map_projection = st.selectbox("Map Perspective:", ["natural earth", "orthographic (3D Globe)", "mercator"])
    with map_c2:
        st.caption("Tip: Hover over circles to inspect company valuation, lead investors, and headquarters city. Notice the massive cluster in San Francisco!")

    # Sort so smallest circles are rendered first and massive titans render on top
    map_data = filtered_df.sort_values(by="valuation_b", ascending=True).copy()

    fig_map = px.scatter_geo(
        map_data,
        lat="latitude",
        lon="longitude",
        hover_name="Company",
        size="map_bubble_size",
        color="clean_industry",
        hover_data={
            "valuation_b": ":$.1fB",
            "city": True,
            "country": True,
            "year_joined": True,
            "select_investors": True,
            "map_bubble_size": False,
            "latitude": False,
            "longitude": False
        },
        size_max=46,
        projection="orthographic" if "3D" in map_projection else map_projection,
        template="plotly_dark",
        color_discrete_sequence=px.colors.qualitative.Prism
    )
    fig_map.update_geos(
        showcoastlines=True, coastlinecolor="#334155",
        showland=True, landcolor="#0F172A",
        showocean=True, oceancolor="#090D16",
        showcountries=True, countrycolor="#1E293B"
    )
    fig_map.update_layout(
        margin={"r": 0, "t": 20, "l": 0, "b": 0},
        height=560,
        legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5)
    )
    st.plotly_chart(fig_map, use_container_width=True)

    c_left, c_right = st.columns(2)
    with c_left:
        st.write("##### Total Valuation by Sector ($B)")
        sector_agg = filtered_df.groupby("clean_industry")["valuation_b"].sum().reset_index().sort_values(by="valuation_b", ascending=True)
        fig_sec = px.bar(
            sector_agg,
            x="valuation_b",
            y="clean_industry",
            orientation="h",
            labels={"valuation_b": "Total Market Cap ($B)", "clean_industry": ""},
            template="plotly_dark",
            color="valuation_b",
            color_continuous_scale="Teal"
        )
        fig_sec.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0), coloraxis_showscale=False)
        st.plotly_chart(fig_sec, use_container_width=True)

    with c_right:
        st.write("##### Top 10 Global Cities by Unicorn Count")
        city_counts = filtered_df["city"].value_counts().head(10).reset_index()
        city_counts.columns = ["City", "Count"]
        fig_city = px.bar(
            city_counts.sort_values(by="Count", ascending=True),
            x="Count",
            y="City",
            orientation="h",
            labels={"Count": "Active Unicorns", "City": ""},
            template="plotly_dark",
            color="Count",
            color_continuous_scale="Purp"
        )
        fig_city.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0), coloraxis_showscale=False)
        st.plotly_chart(fig_city, use_container_width=True)

# =====================================================================
# TAB 2: The VC Valuation Simulator
# =====================================================================
with tab2:
    st.subheader("Interactive Valuation & Cap Table Simulator")
    st.markdown("""
    Select startup attributes below to predict the **fair-market post-money valuation** using our trained **Random Forest Regressor** and compare against comps like Anthropic, OpenAI, Stripe, and Databricks.
    """)

    sim_col1, sim_col2 = st.columns([1, 1.3], gap="large")

    with sim_col1:
        st.markdown("#### 1. Configure Startup Profile")
        sim_name = st.text_input("Startup / Project Name:", value="Nexus AI")
        
        sim_ind = st.selectbox(
            "Primary Industry Sector:",
            sorted(df["clean_industry"].unique().tolist()),
            index=sorted(df["clean_industry"].unique().tolist()).index("Enterprise Tech") if "Enterprise Tech" in df["clean_industry"].unique() else 0
        )

        sim_hub = st.selectbox(
            "Headquarters Hub Category:",
            sorted(df["hub_category"].unique().tolist()),
            index=sorted(df["hub_category"].unique().tolist()).index("SF Bay Area") if "SF Bay Area" in df["hub_category"].unique() else 0
        )

        sim_tenure = st.slider(
            "Company Operating Horizon / Maturity (Years):",
            min_value=0.5,
            max_value=15.0,
            value=3.0,
            step=0.5,
            help="Years active operating and scaling."
        )

        sim_ai = st.checkbox("Frontier AI / Foundation Model / DeepTech Company", value=True)

        sim_investors = st.slider("Institutional Investors Count:", min_value=1, max_value=15, value=4)

        st.markdown("##### Tier-1 VC Backing On Cap Table:")
        b_c1, b_c2 = st.columns(2)
        with b_c1:
            has_seq = st.checkbox("Sequoia Capital", value=True)
            has_a16 = st.checkbox("Andreessen Horowitz (a16z)", value=False)
            has_goog = st.checkbox("Google / GV / CapitalG", value=False)
            has_khos = st.checkbox("Khosla Ventures", value=True)
        with b_c2:
            has_thriv = st.checkbox("Thrive Capital", value=False)
            has_soft = st.checkbox("SoftBank Vision Fund", value=False)
            has_acc = st.checkbox("Accel", value=False)
            has_tig = st.checkbox("Tiger Global", value=False)

    with sim_col2:
        st.markdown(f"#### 2. Valuation Inference for **{sim_name}**")

        pred_res = predict_valuation(
            tenure_years=sim_tenure,
            investors_count=sim_investors,
            industry=sim_ind,
            hub_category=sim_hub,
            is_ai_frontier=int(sim_ai),
            has_sequoia=int(has_seq),
            has_a16z=int(has_a16),
            has_google=int(has_goog),
            has_khosla=int(has_khos),
            has_thrive=int(has_thriv),
            has_softbank=int(has_soft),
            has_accel=int(has_acc),
            has_tiger=int(has_tig)
        )

        pred_val_m = pred_res["pred_valuation_m"]
        pred_val_b = pred_res["pred_valuation_b"]
        ci_lower_b = pred_res["ci_lower_b"]
        ci_upper_b = pred_res["ci_upper_b"]

        status_text = "🏆 Mega-Decacorn ($10B+)" if pred_val_b >= 10.0 else ("🦄 Unicorn Status ($1B+)" if pred_val_b >= 1.0 else "🌱 Scaling Stage (<$1B)")
        status_color = "#10B981" if pred_val_b >= 10.0 else ("#38BDF8" if pred_val_b >= 1.0 else "#F59E0B")

        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1E293B, #0F172A); padding: 22px; border-radius: 12px; border: 1px solid #38BDF8; box-shadow: 0 4px 20px rgba(56, 189, 248, 0.2);">
            <div style="color: #38BDF8; font-weight: 600; text-transform: uppercase; font-size: 0.85rem; letter-spacing: 0.5px;">Model Predicted Fair-Market Valuation</div>
            <div style="font-size: 2.8rem; font-weight: 800; color: #FFFFFF; margin: 4px 0;">
                ${pred_val_b:.2f} Billion <span style="font-size: 1.2rem; color: #94A3B8;">(${pred_val_m:,.0f}M)</span>
            </div>
            <div style="color: #94A3B8; font-size: 0.9rem;">
                Estimated 80% Confidence Interval: <strong style="color: #F8FAFC;">${ci_lower_b:.2f}B – ${ci_upper_b:.2f}B</strong>
            </div>
            <hr style="border-color: rgba(255,255,255,0.1); margin: 16px 0;">
            <div style="display: flex; justify-content: space-between; font-size: 0.95rem;">
                <div>Cohort Classification: <strong style="color: {status_color};">{status_text}</strong></div>
                <div>AI Frontier Premium Applied: <strong style="color: {'#10B981' if sim_ai else '#94A3B8'};">{'✅ Active (+78.6%)' if sim_ai else '❌ Inactive'}</strong></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.write("")
        st.markdown("##### 📌 Nearest Real-World Unicorn Comps")
        comps = find_closest_comps(df, sim_ind, sim_hub, sim_ai, n=3)
        for _, comp in comps.iterrows():
            st.markdown(f"""
            <div class="comp-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div class="comp-title">{comp['Company']}</div>
                    <div style="font-weight: 700; color: #10B981; font-size: 1.15rem;">${comp['valuation_b']:.1f}B Valuation</div>
                </div>
                <div style="color: #94A3B8; font-size: 0.85rem; margin-top: 4px;">
                    📍 {comp['city']}, {comp['country']} | 🗓️ Joined: {comp['year_joined']} | ⚡ Velocity: ${comp['valuation_velocity_b']:.1f}B/yr
                </div>
                <div style="color: #64748B; font-size: 0.8rem; margin-top: 3px;">
                    Lead Investors: {comp['select_investors']}
                </div>
            </div>
            """, unsafe_allow_html=True)

# =====================================================================
# TAB 3: Velocity & Decacorn Screener
# =====================================================================
with tab3:
    st.subheader("Valuation Velocity & Decacorn Alpha Screener")
    st.markdown("""
    By analyzing **Unicorn Tenure (Years)** against **Post-Money Valuation ($B)** on a logarithmic scale, we detect the **Decacorn Alpha** cohort—companies that scaled to tens or hundreds of billions in record time.
    """)

    fig_scatter = px.scatter(
        df,
        x="tenure_years",
        y="valuation_b",
        color="efficiency_tier",
        hover_name="Company",
        log_y=True,
        hover_data={
            "valuation_b": ":$.1fB",
            "year_joined": True,
            "valuation_velocity_b": ":$.1fB/yr",
            "clean_industry": True,
            "city": True,
            "country": True
        },
        labels={
            "tenure_years": "Operating Horizon as Unicorn (Years)",
            "valuation_b": "Post-Money Valuation ($B, Log Scale)",
            "efficiency_tier": "Valuation Classification"
        },
        color_discrete_map={
            "Decacorn Alpha (Exceeds Market Trajectory)": "#10B981",
            "Fair Market Valuation": "#38BDF8",
            "Valuation Discounted / Stagnant": "#F43F5E"
        },
        template="plotly_dark"
    )
    fig_scatter.update_layout(
        height=520,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    col_alpha, col_guzz = st.columns(2)
    with col_alpha:
        st.write("##### 🚀 Top 10 Highest Valuation Velocity Titans")
        st.caption("Companies creating the largest annual enterprise valuation since inception.")
        top_alpha = df.sort_values(by="valuation_velocity_b", ascending=False)[
            ["Company", "clean_industry", "valuation_b", "year_joined", "valuation_velocity_b"]
        ].head(10)
        top_alpha.columns = ["Company", "Industry", "Valuation ($B)", "Year Joined", "Velocity ($B/yr)"]
        st.dataframe(top_alpha.style.format({
            "Valuation ($B)": "${:.1f}B",
            "Velocity ($B/yr)": "${:.1f}B/yr"
        }), use_container_width=True)

    with col_guzz:
        st.write("##### 🌐 Top Frontier AI & Machine Learning Leaders")
        st.caption("The vanguard of foundation model creators and generative AI platforms.")
        top_ai = df[df["is_ai_frontier"] == 1].sort_values(by="valuation_b", ascending=False)[
            ["Company", "clean_industry", "valuation_b", "city", "country"]
        ].head(10)
        top_ai.columns = ["Company", "Industry", "Valuation ($B)", "City", "Country"]
        st.dataframe(top_ai.style.format({
            "Valuation ($B)": "${:.1f}B"
        }), use_container_width=True)

# =====================================================================
# TAB 4: Model Diagnostics & Elasticity
# =====================================================================
with tab4:
    st.subheader("Statistical Diagnostics, Benchmark & VC Elasticity")
    st.markdown("""
    Empirical evaluation of factors driving private tech valuations across 1,447 global unicorns.
    """)

    b1, b2, b3 = st.columns(3)
    with b1:
        st.metric("Total Sample Size", f"{len(df):,} Unicorns", "+400 modern additions")
    with b2:
        st.metric("Test Set Log R² Score", f"{bundle['test_r2']:.4f}", "Cross-validated holdout")
    with b3:
        st.metric("Selected Model Architecture", bundle['model_name'], "Non-linear tree ensemble")

    st.write("---")
    st.write("##### 🎯 Feature Elasticity & Valuation Premiums (Ridge Regression)")
    st.markdown("Percentage change in expected post-money valuation associated with each attribute, holding other variables constant.")

    coef_df = bundle["coefficients"].copy()
    coef_df["Display Name"] = (
        coef_df["feature"]
        .str.replace("clean_industry_", "Sector: ")
        .str.replace("hub_category_", "Hub: ")
        .str.replace("has_", "Backer: ")
        .str.replace("is_ai_frontier", "⚡ Frontier AI Status")
        .str.replace("tenure_years", "⏳ Tenure (Years)")
        .str.replace("investors_count", "🤝 Investor Count")
    )

    fig_coef = px.bar(
        coef_df.sort_values(by="coefficient", ascending=True),
        x="coefficient",
        y="Display Name",
        orientation="h",
        color="coefficient",
        color_continuous_scale="Spectral",
        labels={"coefficient": "Log-Scale Elasticity (Beta Coefficient)", "Display Name": ""},
        template="plotly_dark"
    )
    fig_coef.update_layout(height=520, margin=dict(l=0, r=0, t=10, b=0), coloraxis_showscale=False)
    st.plotly_chart(fig_coef, use_container_width=True)

    with st.expander("📚 Methodology & Academic Defense Notes (Click to expand)"):
        st.markdown(r"""
        1. **Power-Law Dynamics & Log-Scale Modeling:**
           Startup valuations exhibit heavy-tailed **Pareto / Power-Law distributions** where decacorns and centicorns (OpenAI, Anthropic, ByteDance) account for a disproportionate share of global value. Modeling $\log(\text{Valuation})$ stabilizes variance, linearizes multiplicative returns, and ensures positive predictions.
           
        2. **Frontier AI Multiple Expansion:**
           Empirical regression demonstrates an estimated **+78.6% valuation premium** associated with frontier AI positioning over equivalent traditional SaaS companies, reflecting extreme investor demand for foundation model capabilities.
           
        3. **Investor Network Capital:**
           Having Tier-1 venture capital backing (Sequoia, Google, SoftBank, a16z, Khosla) commands a **40% to 77% valuation multiplier**, capturing signal value, syndicate power, and competitive deal heat.
        """)
