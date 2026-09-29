# Building Heating Load Predictor

A Flask web app that predicts building heating load from eight building characteristics using a trained linear regression model.

## Features

- Relative Compactness
- Surface Area
- Wall Area
- Roof Area
- Overall Height
- Orientation
- Glazing Area
- Glazing Area Distribution

The model output is **Y1 Heating Load**.

## Run locally

1. Create and activate a virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

3. Start the app:

   ```powershell
   python app.py
   ```

4. Open http://127.0.0.1:5000/ in your browser.

The model files `linear_regression_model.pkl` and `scaler.pkl` are included because the application loads them at startup.
