"""
==============================================================================
 app.py — Flask Web Application  (Backend Server)
==============================================================================
 MODULE OVERVIEW  (Viva Explanation)
 ───────────────────────────────────
 This file is the **web server**.  It uses the Flask micro-framework
 to serve two kinds of things:

   1. PAGE ROUTE  (GET /)
      Returns the HTML dashboard page to the user's browser.

   2. API ENDPOINTS  (GET/POST /api/…)
      Return JSON data that the frontend JavaScript fetches
      asynchronously (AJAX) and renders as interactive charts.

 ROUTES
 ──────
   GET  /                → Dashboard page
   POST /api/train       → Train the ML model
   GET  /api/predict     → 7-day hourly predictions
   GET  /api/daily       → Daily usage data
   GET  /api/hourly      → 24-hour profile
   GET  /api/weekly      → Day-of-week averages
   GET  /api/monthly     → Monthly totals
   GET  /api/summary     → Summary statistics
   GET  /api/recommend   → AI energy-saving tips
   GET  /api/metrics     → Model evaluation metrics
   GET  /api/importance  → Feature importance scores
   GET  /api/anomalies   → Anomaly detection results
   GET  /api/appliances  → Appliance-level breakdown
   POST /upload          → Upload a custom dataset CSV
==============================================================================
"""

# ── Imports ──────────────────────────────────────────────────────────────
import os
from flask import Flask, render_template, request, jsonify, send_file

import model as ml
from generate_dataset import generate_dataset

# ── Flask Setup ──────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024   # 200 MB upload limit

DATASET_PATH = "dataset.csv"

# Auto-generate dataset on first run
if not os.path.exists(DATASET_PATH):
    print("📊  No dataset found — generating synthetic data …")
    generate_dataset(DATASET_PATH)


# ═══════════════════════════════════════════════════════════════════════════
#  PAGE ROUTE
# ═══════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    """Serve the dashboard page."""
    return render_template("index.html", model_ready=ml.is_model_trained())


# ═══════════════════════════════════════════════════════════════════════════
#  API ENDPOINTS  (all return JSON)
# ═══════════════════════════════════════════════════════════════════════════

@app.route("/api/train", methods=["POST"])
def api_train():
    """Train the ML model and return evaluation metrics."""
    try:
        metrics = ml.train_model(DATASET_PATH)
        return jsonify({"status": "success", "metrics": metrics})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/predict")
def api_predict():
    """Predict next 7 days of hourly energy consumption."""
    if not ml.is_model_trained():
        return jsonify({"status": "error",
                        "message": "Model not trained yet."}), 400
    try:
        preds = ml.predict_future(days=7, filepath=DATASET_PATH)
        return jsonify({"status": "success", "predictions": preds})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/daily")
def api_daily():
    return jsonify(ml.get_daily_usage(DATASET_PATH))


@app.route("/api/hourly")
def api_hourly():
    return jsonify(ml.get_hourly_profile(DATASET_PATH))


@app.route("/api/weekly")
def api_weekly():
    return jsonify(ml.get_weekly_usage(DATASET_PATH))


@app.route("/api/monthly")
def api_monthly():
    return jsonify(ml.get_monthly_usage(DATASET_PATH))


@app.route("/api/summary")
def api_summary():
    return jsonify(ml.get_summary_statistics(DATASET_PATH))


@app.route("/api/recommend")
def api_recommend():
    return jsonify(ml.get_recommendations(DATASET_PATH))


@app.route("/api/metrics")
def api_metrics():
    m = ml.get_model_metrics()
    if m:
        return jsonify({"status": "success", "metrics": m})
    return jsonify({"status": "error",
                    "message": "Train the model first."}), 400


@app.route("/api/importance")
def api_importance():
    """Feature importance scores for the trained model."""
    imp = ml.get_feature_importance()
    if imp:
        return jsonify({"status": "success", "importance": imp})
    return jsonify({"status": "error",
                    "message": "Train the model first."}), 400


@app.route("/api/anomalies")
def api_anomalies():
    """Run anomaly detection and return flagged records."""
    try:
        result = ml.detect_anomalies(DATASET_PATH)
        return jsonify({"status": "success", **result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/appliances")
def api_appliances():
    """Appliance-level energy breakdown."""
    try:
        result = ml.get_appliance_breakdown(DATASET_PATH)
        return jsonify({"status": "success", **result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/upload", methods=["POST"])
def upload_dataset():
    """Upload a custom CSV, TXT, or PDF dataset."""
    if "file" not in request.files:
        return jsonify({"status": "error", "message": "No file provided."}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"status": "error", "message": "Empty filename."}), 400

    # Debug: Print file size and Flask limit
    file.seek(0, 2)  # Seek to end
    file_length = file.tell()
    file.seek(0)     # Reset to start
    print(f"[UPLOAD DEBUG] Uploaded file size: {file_length} bytes")
    print(f"[UPLOAD DEBUG] Flask MAX_CONTENT_LENGTH: {app.config.get('MAX_CONTENT_LENGTH')} bytes")

    filename = file.filename.lower()
    import pandas as pd
    import io

    # Helper function to process generic DataFrame parsing and saving
    def _process_and_save_df(df):
        # If 'datetime' column is missing but 'Date' and 'Time' exist, combine them
        if 'datetime' not in df.columns:
            date_col = None
            time_col = None
            for col in df.columns:
                if col.strip().lower() == 'date':
                    date_col = col
                if col.strip().lower() == 'time':
                    time_col = col
            if date_col and time_col:
                df['datetime'] = df[date_col].astype(str) + ' ' + df[time_col].astype(str)
                # Optional: Move 'datetime' to first column
                cols = ['datetime'] + [c for c in df.columns if c != 'datetime']
                df = df[cols]

        max_rows = 150000
        if len(df) > max_rows:
            step = len(df) // max_rows
            df = df.iloc[::step].copy()
            msg = f"Dataset was very large. Sampled {len(df)} rows across the entire timeline for efficient AI training."
        else:
            msg = f"Dataset uploaded successfully! ({len(df)} records, {len(df.columns)} features)"
        # Save as CSV (comma-separated)
        df.to_csv(DATASET_PATH, index=False)
        return jsonify({"status": "success", "message": msg})

    if filename.endswith(".csv") or filename.endswith(".txt"):
        try:
            content = file.read().decode('utf-8')
            df = pd.read_csv(io.StringIO(content), sep=',')
            if len(df.columns) <= 1:
                df = pd.read_csv(io.StringIO(content), sep=';')
            if len(df.columns) <= 1:
                df = pd.read_csv(io.StringIO(content), sep='\t')
            if len(df.columns) <= 1:
                df = pd.read_csv(io.StringIO(content), sep=r'\s+')
            return _process_and_save_df(df)
        except Exception as e:
            return jsonify({"status": "error", "message": f"Failed to parse file: {str(e)}"}), 400
            
    elif filename.endswith(".pdf"):
        try:
            import pdfplumber
            # Save the uploaded file temporarily
            tmp_path = "temp_upload.pdf"
            file.save(tmp_path)
            
            all_tables = []
            with pdfplumber.open(tmp_path) as pdf:
                for page in pdf.pages:
                    table = page.extract_table()
                    if table:
                        all_tables.extend(table)
            os.remove(tmp_path)
            
            if not all_tables or len(all_tables) < 2:
                return jsonify({"status": "error", "message": "No valid data table found in PDF."}), 400
                
            # Assume first row is header
            headers = all_tables[0]
            data = all_tables[1:]
            df = pd.DataFrame(data, columns=headers)
            return _process_and_save_df(df)
            
        except Exception as e:
            return jsonify({"status": "error", "message": f"Failed to parse PDF file: {str(e)}"}), 400

    return jsonify({"status": "error",
                    "message": "Only .csv, .txt, and .pdf files are accepted."}), 400


# ─── NEW: Advanced Forecast (LSTM / Holt-Winters) ────────────────────────

@app.route("/api/forecast", methods=["POST"])
def api_forecast():
    """Train LSTM / Holt-Winters forecast and return 14-day prediction."""
    try:
        result = ml.train_forecast_model(DATASET_PATH)
        return jsonify({"status": "success", **result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/forecast/result")
def api_forecast_result():
    """Load saved forecast result (no re-training)."""
    result = ml.get_forecast_result()
    if result:
        return jsonify({"status": "success", **result})
    return jsonify({"status": "error",
                    "message": "Run forecast first."}), 400


# ─── NEW: Trend Analysis ─────────────────────────────────────────────────

@app.route("/api/trends")
def api_trends():
    """Multi-scale energy trend analysis."""
    try:
        result = ml.get_trend_analysis(DATASET_PATH)
        return jsonify({"status": "success", **result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ─── NEW: AI Insights ────────────────────────────────────────────────────

@app.route("/api/insights")
def api_insights():
    """Generate natural-language AI insights."""
    try:
        insights = ml.generate_ai_insights(DATASET_PATH)
        return jsonify({"status": "success", "insights": insights})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ─── NEW: Threshold Alerts ───────────────────────────────────────────────

@app.route("/api/alerts")
def api_alerts():
    """Check energy thresholds and return alerts."""
    try:
        dt = request.args.get("daily_threshold", None, type=float)
        ht = request.args.get("hourly_threshold", None, type=float)
        result = ml.check_alerts(DATASET_PATH, dt, ht)
        return jsonify({"status": "success", **result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ─── NEW: PDF Report Download ────────────────────────────────────────────

@app.route("/api/report")
def api_report():
    """Generate and download a PDF energy report."""
    try:
        path = ml.generate_pdf_report(DATASET_PATH)
        return send_file(path, as_attachment=True,
                         download_name="EnergyAI_Report.pdf",
                         mimetype="application/pdf")
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ─── NEW: Tips Endpoint (alias for recommendations) ──────────────────────

@app.route("/api/tips")
def api_tips():
    """Return energy-saving tips."""
    return jsonify(ml.get_recommendations(DATASET_PATH))


# ═══════════════════════════════════════════════════════════════════════════
#  RUN SERVER
# ═══════════════════════════════════════════════════════════════════════════

# ─── GLOBAL ERROR HANDLERS ──────────────────────────────────────────────
from flask import make_response
from werkzeug.exceptions import HTTPException

@app.errorhandler(HTTPException)
def handle_http_exception(e):
    response = e.get_response()
    # Replace the body with JSON
    response.data = jsonify({
        "status": "error",
        "code": e.code,
        "message": e.description
    }).data
    response.content_type = "application/json"
    return response

@app.errorhandler(Exception)
def handle_exception(e):
    # For any non-HTTPException, return JSON with 500 status
    return make_response(jsonify({
        "status": "error",
        "code": 500,
        "message": str(e)
    }), 500)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
