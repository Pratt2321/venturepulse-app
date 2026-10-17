# VenturePulse: Global Startup Valuation & Market Intelligence

> **Reverse-engineering private tech valuations across 1,445 global unicorns (including OpenAI, Anthropic, DeepSeek, and ByteDance) using empirical ML regression and investor network elasticity.**

---

## Executive Summary & Research Question
Unlike public equities where market pricing occurs continuously with transparent price discovery, private tech valuations are historically opaque as they are negotiated behind closed doors.

This project analyzes the latest verified cohort of **1,445 global tech unicorns** (representing over **$7.36 Trillion** in total enterprise valuation) to answer three core empirical questions:
1. **The Valuation Equation:** *To what extent can a startup's post-money valuation be quantitatively explained by investor network capital, sector, geography, operating tenure, and frontier AI positioning?*
2. **The Frontier AI Premium:** *Holding operating maturity and geography constant, what is the statistical valuation premium commanded by foundation model creators and generative AI platforms over traditional software?*
3. **Decacorn Velocity:** *Which companies represent true hyper-scaling outliers (e.g. Anthropic, OpenAI, DeepSeek, Cognition) that scaled to tens or hundreds of billions in record time?*

---

## Dataset Provenance & Preprocessing
* **Primary Source:** Verified dataset derived from **CB Insights Global Unicorn Club** (`cbinsights_unicorns.csv`), tracking active private companies globally valued at $\ge \$1\text{B}$.
* **Scope & Deduplication:** 1,447 raw rows cleaned and deduplicated to **1,445 unique unicorns** across 55 countries, featuring frontier tech leaders:
  * **Anthropic** ($965B)
  * **OpenAI** ($840B)
  * **ByteDance** ($480B)
  * **Databricks** ($190B)
  * **Stripe** ($159B)
  * **VAST Data** ($80B)
  * **Revolut** ($75B)
  * **DeepSeek** ($50B)
  * **Moonshot AI** ($50B)
  * **Cognition** ($48B)
* **Feature Engineering & Data Integrity:**
  * Cleaned valuations and extracted log-scale targets ($\text{Median Valuation} = \$1.70\text{B}$).
  * Mapped geocoding across 320+ global cities and tech hubs.
  * Extracted Tier-1 investor signals (`has_sequoia`, `has_a16z`, `has_google`, `has_khosla`, `has_thrive`, `has_softbank`, `has_accel`, `has_tiger`).
  * Engineered frontier AI indicators (`is_ai_frontier`) and annual valuation velocity metrics.

---

## Modeling Methodology & Academic Findings

### 1. Power-Law / Log-Scale Transformation
Venture-backed valuations follow an extreme **Pareto / Power-Law Distribution** (skewness $> 8.0$). Modeling $\log(\text{Valuation})$ stabilizes variance, linearizes multiplicative returns, and guarantees strictly positive valuation predictions.

### 2. Empirical Valuation Drivers (Ridge Elasticity)
* **Frontier AI Status:** **$+79.4\%$** valuation premium over equivalent non-AI enterprise software.
* **Sequoia Capital Backing:** **$+76.8\%$** valuation premium.
* **Google / CapitalG / GV Backing:** **$+51.4\%$** valuation multiplier.
* **SoftBank Vision Fund:** **$+50.3\%$** valuation multiplier.
* **Andreessen Horowitz (a16z):** **$+48.4\%$** valuation multiplier.
* **Accel Partners:** **$+47.1\%$** valuation multiplier.
* **Thrive Capital:** **$+40.3\%$** valuation multiplier.
* **SF Bay Area Location:** **$+36.2\%$** valuation uplift over other domestic and international tech hubs (376 unicorns across the Bay Area regional cluster, with 229 in San Francisco proper).

---

## Key Features of the Streamlit Application
1. **Interactive Global Map & Macro Landscape:**
   * Geospatial natural earth scatter map of all 1,445 unicorns worldwide.
   * Unified 6-industry dark-mode palette (Electric Emerald, Bright Cyan, Coral Pink, Vibrant Purple, Gold/Amber, Muted Slate Grey) with 80% opacity fills and crisp halo borders.
   * Linked visual breakdown for **Total Valuation by Sector ($B)** and **Top 10 Global Cities by Unicorn Count** spotlighting Silicon Valley concentration.
2. **The VC Valuation Simulator:** Real-time inference engine powered by our trained **Random Forest Regressor** where users select sector, hub location, company maturity, AI frontier status, and investor backing to compute:
   * Fair-Market Valuation with an **80% Confidence Interval**.
3. **Statistical Diagnostics & Elasticity:** 5-fold cross-validation metrics, Ridge elasticity coefficients, and academic defense methodology notes.

---

## Local Quick Start

### 1. Clone & Setup
```bash
git clone https://github.com/Pratt2321/venturepulse-app.git
cd venturepulse-app
pip install -r requirements.txt
```

### 2. Run Pipeline (Optional — Pre-trained Bundle Included)
```bash
python3 data/download_and_clean.py
python3 models/train_model.py
```

### 3. Launch the Streamlit App
```bash
streamlit run app/app.py
```
