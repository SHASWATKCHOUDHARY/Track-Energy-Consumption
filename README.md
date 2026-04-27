# ⚡ Smart AI Energy Monitoring System

> **INT 428 — Artificial Intelligence Essentials**  
> A production-grade AI-powered dashboard that monitors household energy, forecasts demand with LSTM / Holt-Winters, detects anomalies, generates alerts, produces natural-language insights, and exports automated PDF reports.

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Features](#-features)
3. [Folder Structure](#-folder-structure)
4. [Tech Stack](#-tech-stack)
5. [Setup Instructions](#-setup-instructions)
6. [How to Run & Demo](#-how-to-run--demo)
7. [Module-Wise Explanation (Viva Guide)](#-module-wise-explanation-viva-guide)
8. [AI Techniques Used](#-ai-techniques-used)
9. [API Endpoints](#-api-endpoints)
10. [Screenshots](#-screenshots)
11. [Future Scope](#-future-scope)
12. [Contributors](#-contributors)

---

## 🎯 Project Overview

Energy consumption in households is rising globally. This project builds a **Smart AI Energy Monitoring System** that:

* **Monitors** hourly energy usage across 6 appliance categories.
* **Forecasts** future demand using **LSTM neural network** (deep learning) or **Holt-Winters** (statistical time-series).
* **Analyses trends** — weekly, monthly, rolling averages, and directional slopes.
* **Detects anomalies** using dual-method AI (Isolation Forest + Z-Score).
* **Generates alerts** when consumption exceeds dynamic thresholds.
* **Produces AI insights** — natural-language sentences like *"Your consumption will increase by 12% next week."*
* **Exports PDF reports** — automated multi-page energy report with one click.
* **Recommends** energy-saving actions with priority levels.
* Supports **Dark / Light theme** toggle for presentation flexibility.

---

## ✨ Features

| Feature | Description |
|---|---|
| 📊 **Dashboard Overview** | 8 stat cards — total kWh, avg daily, peak hour/day/month, std deviation |
| 📈 **Interactive Charts** | Plotly.js — daily, hourly, weekly, monthly, all zoomable & hoverable |
| 📉 **Trend Analysis** | 7-day rolling average, monthly % change, overall direction (slope) |
| 🏠 **Appliance Breakdown** | Pie + stacked bar: AC, Lighting, Kitchen, Fan, Entertainment, Other |
| 🚨 **Anomaly Detection** | Isolation Forest + Z-Score dual-method, with scatter chart & table |
| 🔔 **Threshold Alerts** | Dynamic alerts with configurable daily/hourly thresholds, severity levels |
| 🧠 **LSTM / Holt-Winters Forecast** | 14-day advanced time-series prediction with R²/RMSE/MAE metrics |
| 🤖 **RF 7-Day Prediction** | Hourly 168-point forecast using Random Forest regressor |
| 💡 **AI Insights** | Natural-language analysis: trends, costs, peaks, appliances, anomalies |
| 📄 **PDF Report** | One-click downloadable report with stats, trends, insights, alerts, tips |
| 🌿 **Smart Recommendations** | Priority-tagged tips (HIGH/MEDIUM/LOW) with estimated savings % |
| ⚙️ **Model Comparison** | Side-by-side LR vs RF with MAE, RMSE, R², MAPE + Feature Importance |
| 🌗 **Dark / Light Theme** | CSS-variable powered toggle with localStorage persistence |
| 📁 **CSV Upload** | Upload custom datasets and retrain models on-the-fly |

---

## 📂 Folder Structure

```
INT 428 Project/
│
├── app.py                  # Flask server (17 API endpoints)
├── model.py                # ML pipeline (13 sections — training, LSTM, insights, PDF)
├── generate_dataset.py     # Synthetic data generator (appliances + anomalies)
├── dataset.csv             # 8,760 hourly records (auto-generated)
├── requirements.txt        # Python dependencies
├── README.md               # This file
│
├── templates/
│   └── index.html          # Dashboard frontend (12 sidebar sections + Plotly.js)
│
├── static/
│   └── style.css           # Dual-theme CSS (dark/light)
│
└── saved_models/           # Created after training
    ├── energy_model.pkl    # Random Forest model
    ├── scaler.pkl          # Feature scaler
    ├── feature_importance.json
    ├── metrics.json        # RF/LR evaluation metrics
    ├── forecast_result.json # LSTM/Holt-Winters forecast
    └── energy_report.pdf   # Auto-generated PDF report
```

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.10+ |
| **Web Framework** | Flask 3.0 |
| **ML (Supervised)** | Scikit-learn — RandomForest, LinearRegression |
| **ML (Unsupervised)** | Scikit-learn — IsolationForest |
| **Time-Series** | statsmodels — Holt-Winters Exponential Smoothing |
| **Deep Learning** | TensorFlow/Keras — LSTM *(optional, falls back to Holt-Winters)* |
| **PDF Generation** | fpdf2 |
| **Data Processing** | NumPy, Pandas |
| **Charting** | Plotly.js 2.27 (interactive, theme-aware) |
| **Frontend** | HTML5, CSS3 (custom properties), JavaScript |
| **Data Format** | CSV |

---

## 🚀 Setup Instructions

### Prerequisites

* **Python 3.10+** installed
* **pip** (included with Python)
* A modern web browser (Chrome, Edge, Firefox)

### Step-by-Step

```bash
# 1. Open terminal in the project folder
cd "INT 428 Project"

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Install TensorFlow for LSTM support
pip install tensorflow

# 5. Generate the synthetic dataset
python generate_dataset.py

# 6. Train all models (RF + Holt-Winters/LSTM + generate PDF)
python model.py

# 7. Start the web application
python app.py
```

Open **http://localhost:5000** in your browser.

---

## ▶️ How to Run & Demo

| Step | Action | What Happens |
|---|---|---|
| 1 | `python app.py` | Flask server starts at port 5000 |
| 2 | Open dashboard | Overview stats, charts, trends, and insights load automatically |
| 3 | Click **🧠 Train Model** | Trains Random Forest + Linear Regression, shows metrics |
| 4 | Click **🚀 Run Forecast** | Runs LSTM or Holt-Winters 14-day forecast |
| 5 | Click **🔮 Load Prediction** | Shows RF 7-day hourly prediction |
| 6 | Click **🔎 Detect Anomalies** | Runs Isolation Forest + Z-Score |
| 7 | Click **🔔 Check Alerts** | Generates threshold-based alerts |
| 8 | Click **📄 Report** | Downloads a multi-page PDF energy report |
| 9 | Click **🌓** | Toggle dark / light theme |

---

## 🧠 Module-Wise Explanation (Viva Guide)

### Module 1: `generate_dataset.py` — Data Generation

**Purpose:** Creates realistic synthetic household energy data.

* 8,760 records (365 days × 24 hours) starting 2025-01-01
* 6 appliance categories with time-of-day, seasonal, and weekend patterns
* ~1.5% anomaly injection (2×–4× spikes) simulating real-world events
* `energy_kwh = sum(ac + lighting + kitchen + fan + entertainment + other)`

---

### Module 2: `model.py` — ML Pipeline (13 Sections)

| Section | Function | AI Technique |
|---|---|---|
| 1. Data Loading | `load_data()` | Pandas I/O |
| 2. Feature Engineering | `engineer_features()` | Cyclical sin/cos encoding |
| 3. Model Training | `train_model()` | Supervised regression (LR + RF) |
| 4. RF Prediction | `predict_future()` | 7-day hourly inference |
| 5. Anomaly Detection | `detect_anomalies()` | IsolationForest + Z-Score |
| 6. Analytics | `get_summary_statistics()` etc. | Aggregation + profiling |
| 7. Recommendations | `get_recommendations()` | Rule-based expert system |
| 8. Utilities | `is_model_trained()`, `get_model_metrics()` | I/O helpers |
| **9. Advanced Forecast** | **`train_forecast_model()`** | **LSTM / Holt-Winters** |
| **10. Trend Analysis** | **`get_trend_analysis()`** | **Rolling avg + linear slope** |
| **11. AI Insights** | **`generate_ai_insights()`** | **NL expert system** |
| **12. Threshold Alerts** | **`check_alerts()`** | **Dynamic threshold monitoring** |
| **13. PDF Report** | **`generate_pdf_report()`** | **Automated report generation** |

---

### Module 3: `app.py` — Flask Server (17 Routes)

Serves the dashboard HTML and 17 RESTful JSON API endpoints (see API table below).

---

### Module 4: `templates/index.html` — Dashboard (12 Sections)

**New sections added:**
* 📉 **Trends** — Rolling average chart + monthly % change bar chart
* 🔔 **Alerts** — Configurable threshold inputs + alert table with severity badges
* 🧠 **Forecast** — LSTM/Holt-Winters 14-day chart + metrics + result table
* 💡 **AI Insights** — Grid of natural-language insight cards with severity colouring

---

### Module 5: `static/style.css` — Dual-Theme Styling

CSS custom properties (`--clr-bg`, `--clr-card`, etc.) switch between dark and light palettes. New classes: `.insight-card`, `.insight-*`, `.alert-controls`, `.input-field`, `.badge-warn`, `.alert-critical`, `.alert-warning`.

---

## 🤖 AI Techniques Used

### 1. LSTM — Long Short-Term Memory (Deep Learning)
* Recurrent Neural Network designed for sequential data
* Maintains memory cells that learn long-term temporal patterns
* Architecture: `LSTM(64) → Dropout(0.2) → LSTM(32) → Dropout(0.2) → Dense(1)`
* 14-day lookback window → predicts next day's total kWh
* **Requires TensorFlow** — falls back to Holt-Winters if not installed

### 2. Holt-Winters Triple Exponential Smoothing
* Classical statistical time-series method from `statsmodels`
* Decomposes series into: Level + Trend + Seasonality
* Uses `seasonal_periods=7` (weekly cycle) with additive components
* Always available as fallback when LSTM isn't feasible

### 3. Random Forest Regressor (Supervised Learning)
* Ensemble of 200 decision trees (max_depth=14)
* Captures non-linear feature interactions
* Evaluated with MAE, RMSE, R², MAPE

### 4. Isolation Forest (Unsupervised Anomaly Detection)
* Randomly partitions data; anomalies are isolated faster (shorter tree paths)
* Combined with Z-Score (rolling 24h window, >3σ) for comprehensive detection

### 5. Trend Analysis (Statistical)
* 7-day rolling average to smooth noise
* Week-over-week and month-over-month % change computation
* Linear regression slope to determine overall direction

### 6. AI Insight Generation (Expert System)
* Rule-based analysis across 7 dimensions: weekly trend, peak hours, appliance share, seasonal forecast, anomaly health, cost estimate, weekend patterns
* Outputs natural-language sentences with severity tags

### 7. Threshold Alert System (IoT Monitoring)
* Dynamic thresholds: daily = mean + 1.5σ, hourly = mean + 2σ
* User-configurable via the dashboard
* Alerts ranked by severity (critical / warning)

### 8. Automated PDF Report (Data Pipeline)
* Collects output from all analysis modules
* Formats into a professional multi-page PDF using fpdf2

---

## 🌐 API Endpoints

| Route | Method | Purpose |
|---|---|---|
| `/` | GET | Dashboard page |
| `/api/summary` | GET | Overview statistics |
| `/api/daily` | GET | Daily usage data |
| `/api/hourly` | GET | 24-hour profile |
| `/api/weekly` | GET | Day-of-week averages |
| `/api/monthly` | GET | Monthly totals |
| `/api/train` | POST | Train RF + LR models |
| `/api/predict` | GET | RF 7-day hourly forecast |
| `/api/metrics` | GET | Model evaluation metrics |
| `/api/importance` | GET | Feature importance scores |
| `/api/anomalies` | GET | Anomaly detection |
| `/api/appliances` | GET | Appliance breakdown |
| `/api/recommend` | GET | Recommendations |
| `/api/forecast` | POST | **Train LSTM/Holt-Winters** |
| `/api/trends` | GET | **Trend analysis** |
| `/api/insights` | GET | **AI insights** |
| `/api/alerts` | GET | **Threshold alerts** |
| `/api/report` | GET | **Download PDF report** |
| `/upload` | POST | Upload custom CSV |

---

### Key Viva Questions & Answers

| # | Question | Answer |
|---|---|---|
| 1 | *What type of ML is this?* | Supervised regression (RF) + Unsupervised anomaly detection (IF) + Deep learning (LSTM) |
| 2 | *What is LSTM?* | Long Short-Term Memory — a type of RNN with memory cells that learn temporal patterns in sequential data. |
| 3 | *Why use Holt-Winters as fallback?* | It's a proven statistical time-series method that captures level + trend + seasonality without needing a GPU or TensorFlow. |
| 4 | *How does Isolation Forest work?* | It randomly partitions data; anomalies are isolated in fewer splits (shorter tree path = more anomalous). |
| 5 | *What is cyclical encoding?* | Converting periodic features (hour, month) to sin/cos so the model knows hour 23 is close to hour 0. |
| 6 | *How does the alert system work?* | It uses dynamic thresholds (mean + k×σ) to flag daily and hourly consumption that exceeds normal ranges. |
| 7 | *What AI generates the insights?* | A rule-based expert system that analyses 7 data dimensions and produces natural-language sentences. |
| 8 | *How is the PDF report generated?* | Using fpdf2 — data from all analysis modules is collected and formatted into a multi-page PDF programmatically. |
| 9 | *Can this work with real data?* | Yes — upload any CSV with the same columns via the dashboard's upload feature. |
| 10 | *What is MAPE?* | Mean Absolute Percentage Error — shows prediction error as a human-readable percentage. |
| 11 | *Why Random Forest over Linear Regression?* | RF handles non-linear relationships and feature interactions; usually achieves higher R². |
| 12 | *What is trend analysis?* | Computing rolling averages and % changes over time to determine if consumption is increasing, decreasing, or stable. |
| 13 | *How is the theme toggle implemented?* | CSS custom properties (`--clr-bg`, etc.) change when a `data-theme` attribute on `<html>` is toggled. Stored in localStorage. |
| 14 | *What makes this a "monitoring system"?* | It combines tracking (charts), forecasting (LSTM/RF), detection (anomalies), alerting (thresholds), and reporting (PDF) — a complete monitoring pipeline. |

---

## 📸 Screenshots

> Run the project and capture screenshots for your report:
> 1. Dashboard overview (dark + light theme)
> 2. Daily usage line chart
> 3. Trend analysis (rolling average + monthly % change)
> 4. Appliance breakdown (pie + stacked bar)
> 5. Anomaly detection results
> 6. Alert system with threshold configuration
> 7. LSTM / Holt-Winters 14-day forecast
> 8. AI Insights cards
> 9. RF 7-day prediction with trend arrows
> 10. Energy-saving recommendations with priority badges
> 11. Model evaluation metrics (LR vs RF)
> 12. Downloaded PDF report

---

## 🔮 Future Scope

1. **Real-Time IoT Integration** — Connect smart meters via MQTT for live tracking.
2. **Deep Learning Expansion** — GRU, Transformer-based time-series models.
3. **User Authentication** — Multi-user dashboards with login.
4. **Cloud Deployment** — Host on AWS / Azure / Heroku.
5. **Cost Prediction** — Integrate local tariff rates for bill estimation.
6. **Solar Integration** — Track generation vs consumption for net-zero.
7. **Reinforcement Learning** — Automated appliance scheduling.
8. **Mobile App** — React Native / Flutter companion.
9. **Real-Time Alerts** — Push notifications via email/SMS.
10. **LLM Reports** — Use GPT to generate human-quality monthly reports.

---

## 👥 Contributors

| Name | Role |
|---|---|
| *Your Name* | Developer & Researcher |
| *Faculty Name* | Project Guide |

---

## 📄 License

This project is developed for academic purposes as part of INT 428 coursework.

---

*Built with ❤️ using Python, Flask, Scikit-learn, statsmodels, and Plotly.js*
