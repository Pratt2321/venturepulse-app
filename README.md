# 🦄 VenturePulse: Global Startup Valuation & Market Intelligence

> **Reverse-engineering private tech valuations across 1,447 global unicorns (including OpenAI, Anthropic, DeepSeek, and ByteDance) using empirical ML regression and investor network elasticity.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E.svg)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Charts-3F4F75.svg)](https://plotly.com/)

---

## 📌 Executive Summary & Research Question
Unlike public equities where market pricing occurs continuously with transparent price discovery, private tech valuations are historically opaque—negotiated behind closed doors in term sheets.

This project analyzes the latest cohort of **1,447 global tech unicorns** (representing over **$7.40 Trillion** in total enterprise valuation) to answer three core empirical questions:
1. **The Valuation Equation:** *To what extent can a startup's post-money valuation be quantitatively explained by investor network capital, sector, geography, operating tenure, and frontier AI positioning?*
2. **The Frontier AI Premium:** *Holding operating maturity and geography constant, what is the statistical valuation premium commanded by foundation model creators and generative AI platforms over traditional software?*
3. **Decacorn Velocity:** *Which companies represent true hyper-scaling outliers (e.g. Anthropic, OpenAI, DeepSeek, Cognition) that scaled to tens or hundreds of billions in record time?*

---

## 📊 Dataset Provenance & Preprocessing
* **Primary Source:** Verified dataset derived from **CB Insights Global Unicorn Club** (`cbinsights_unicorns.csv`), tracking 1,447 active private companies globally valued at $\ge \$1\text{B}$.
* **Scope:** 1,447 unicorns across 55 countries, featuring frontier tech leaders:
  * **Anthropic** ($965B)
  * **OpenAI** ($840B)
  * **ByteDance** ($480B)
  * **Databricks** ($190B)
  * **Stripe** ($159B)
  * **VAST Data** ($80B)
  * **Revolut** ($75B)
  * **DeepSeek** ($50B)
  * **Cognition** ($48B)
  * **Moonshot AI** ($50B)
* **Feature Engineering:**
  * Cleaned valuations and extracted log-scale targets.
  * Standardized geocoding coordinates for 60+ global startup hubs.
  * Extracted Tier-1 investor signals (`has_sequoia`, `has_a16z`, `has_google`, `has_khosla`, `has_thrive`, `has_softbank`, `has_accel`, `has_tiger`).
  * Engineered frontier AI indicators (`is_ai_frontier`) and valuation velocity metrics ($B/year).

---

## 🔬 Modeling Methodology & Academic Findings

### 1. Power-Law / Log-Scale Transformation
Venture-backed valuations follow an extreme **Pareto / Power-Law Distribution** (skewness $> 8.0$). Modeling $\log(\text{Valuation})$ stabilizes variance, linearizes multiplicative returns, and guarantees strictly positive valuation predictions.

### 2. Empirical Valuation Drivers (Ridge Elasticity)
* **⚡ Frontier AI Status:** **$+78.6\%$** valuation premium over equivalent non-AI enterprise software.
* **🌲 Sequoia Capital Backing:** **$+77.3\%$** valuation premium.
* **🌐 Google / CapitalG / GV Backing:** **$+52.3\%$** valuation multiplier.
* **🏦 SoftBank Vision Fund:** **$+50.4\%$** valuation multiplier.
* **🚀 Andreessen Horowitz (a16z):** **$+48.9\%$** valuation multiplier.
* **📍 SF Bay Area Location:** **$+37.7\%$** valuation uplift over other domestic and international tech hubs.

---

## 🚀 Key Features of the Streamlit Application
1. **🗺️ Interactive Global Map:** Geospatial scatter map of all 1,447 global unicorns with bubble sizing by valuation and sector color-coding.
2. **🎛️ The VC Valuation Simulator:** Real-time inference engine where users select sector, hub location, company maturity, AI frontier status, and investor backing to compute:
   * Fair-Market Valuation with an **80% Confidence Interval**.
   * Cohort classification (Mega-Decacorn $\ge \$10\text{B}$, Unicorn $\ge \$1\text{B}$, Scaling Stage).
   * Nearest Real-World Unicorn Comps.
3. **⚡ Valuation Velocity Screener:** Scatter plot analyzing operating tenure vs. post-money valuation, identifying the high-velocity decacorns.
4. **📈 Statistical Diagnostics:** Cross-validation metrics, Ridge elasticity charts, and mathematical justifications.

---

## 💻 Local Quick Start

### 1. Navigate to Project
```bash
cd "/Users/prathampradhan/Developer/CMSE 830/venturepulse-app"
```

### 2. Run Pipeline (Already Pre-Trained & Ready)
```bash
python3 data/download_and_clean.py
python3 models/train_model.py
```

### 3. Launch the Streamlit App
```bash
streamlit run app/app.py
```
