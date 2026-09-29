from pathlib import Path
from textwrap import dedent

import streamlit as st
import joblib
import numpy as np


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Building Energy Model",
    page_icon="🏠",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# ============================================================
# LOAD MODEL
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(BASE_DIR / "linear_regression_model.pkl")
scaler = joblib.load(BASE_DIR / "scaler.pkl")


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    dedent("""
    <style>

    /* --------------------------------------------------------
       GLOBAL
    -------------------------------------------------------- */

    .stApp {
        background: #f4f6ed;
    }

    .main .block-container {
        max-width: 900px;
        padding-top: 35px;
        padding-bottom: 25px;
    }

    /* Hide Streamlit branding */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }


    /* --------------------------------------------------------
       HEADER
    -------------------------------------------------------- */

    .brand {
        font-family: Arial, sans-serif;
        font-size: 8px;
        font-weight: 700;
        letter-spacing: 2px;
        color: #17806f;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .main-title {
        font-family: Georgia, "Times New Roman", serif;
        font-size: 42px;
        line-height: 0.98;
        font-weight: 400;
        color: #071b1a;
        max-width: 550px;
        margin-bottom: 18px;
    }

    .intro {
        font-family: Arial, sans-serif;
        font-size: 8px;
        line-height: 1.5;
        color: #52615e;
        max-width: 560px;
        margin-bottom: 28px;
    }


    /* --------------------------------------------------------
       INPUT CARD
    -------------------------------------------------------- */

    .input-card {
        background: #fffef9;
        border: 1px solid #e0e2d8;
        padding: 22px 22px 20px 22px;
        margin-top: 10px;
    }

    .section-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        border-bottom: 1px solid #d9ddd3;
        padding-bottom: 13px;
        margin-bottom: 14px;
    }

    .section-title {
        display: block;
        font-family: Georgia, "Times New Roman", serif;
        font-size: 17px;
        color: #14201e;
        margin-bottom: 3px;
    }

    .section-subtitle {
        display: block;
        font-family: Arial, sans-serif;
        font-size: 7px;
        color: #6c7773;
    }

    .required {
        display: inline-block;
        background: #edf4c8;
        color: #587c2e;
        font-family: Arial, sans-serif;
        font-size: 6px;
        font-weight: 700;
        letter-spacing: 0.6px;
        padding: 5px 7px;
        margin-top: 1px;
        text-transform: uppercase;
    }


    /* --------------------------------------------------------
       STREAMLIT INPUTS
    -------------------------------------------------------- */

    div[data-testid="stNumberInput"] {
        margin-bottom: 8px;
    }

    div[data-testid="stNumberInput"] label {
        font-family: Arial, sans-serif;
        font-size: 7px !important;
        font-weight: 700 !important;
        letter-spacing: 0.5px;
        color: #18302d !important;
        text-transform: uppercase;
    }

    div[data-testid="stNumberInput"] input {
        background: #fafbf6 !important;
        border: 1px solid #cfd7c8 !important;
        border-radius: 0px !important;
        height: 28px !important;
        font-family: Arial, sans-serif !important;
        font-size: 10px !important;
        color: #20312e !important;
    }

    div[data-testid="stNumberInput"] input:focus {
        border-color: #167b6d !important;
        box-shadow: 0 0 0 1px #167b6d !important;
    }

    div[data-testid="stNumberInput"] button {
        background: transparent !important;
        border: none !important;
    }


    /* --------------------------------------------------------
       INPUT GRID
    -------------------------------------------------------- */

    div[data-testid="column"] {
        padding-left: 4px;
        padding-right: 4px;
    }


    /* --------------------------------------------------------
       BOTTOM INPUT AREA
    -------------------------------------------------------- */

    .bottom-area {
        border-top: 1px solid #d9ddd3;
        margin-top: 8px;
        padding-top: 14px;
    }

    .help-text {
        font-family: Arial, sans-serif;
        font-size: 7px;
        line-height: 1.45;
        color: #65716d;
        max-width: 250px;
    }


    /* --------------------------------------------------------
       BUTTON
    -------------------------------------------------------- */

    div.stButton > button {
        background: #16796c !important;
        color: white !important;
        border: none !important;
        border-radius: 0px !important;
        height: 32px !important;
        font-family: Arial, sans-serif !important;
        font-size: 8px !important;
        font-weight: 700 !important;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        padding: 0px 18px !important;
    }

    div.stButton > button:hover {
        background: #11685d !important;
        color: white !important;
    }

    div.stButton > button:focus {
        box-shadow: none !important;
    }


    /* --------------------------------------------------------
       RESULT CARD
    -------------------------------------------------------- */

    .result-card {
        background: #095b55;
        padding: 18px 18px 20px 18px;
        margin-top: 12px;
        min-height: 125px;
    }

    .result-label {
        display: block;
        font-family: Arial, sans-serif;
        font-size: 7px;
        font-weight: 700;
        letter-spacing: 1px;
        color: #d7ec72;
        text-transform: uppercase;
        margin-bottom: 10px;
    }

    .result-number {
        display: block;
        font-family: Georgia, "Times New Roman", serif;
        font-size: 34px;
        line-height: 1;
        color: #e2f54b;
        margin-bottom: 10px;
    }

    .result-description {
        display: block;
        font-family: Arial, sans-serif;
        font-size: 7px;
        color: #e0ebe7;
        line-height: 1.4;
    }


    /* --------------------------------------------------------
       FOOTER
    -------------------------------------------------------- */

    .footer-row {
        display: flex;
        justify-content: space-between;
        margin-top: 10px;
        font-family: Arial, sans-serif;
        font-size: 6px;
        color: #64706c;
    }

    </style>
    """),
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="brand">BUILDING ENERGY MODEL</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-title">'
    'Estimate the heating<br>load of a building.'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="intro">'
    'Enter the building geometry and glazing details used by the trained model. '
    'Every input is scaled consistently before the heating load is calculated.'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# INPUT CARD HEADER
# ============================================================

st.markdown(
    '<span class="section-title">Building characteristics</span>'
    '<span class="section-subtitle">Enter one numeric value for each model feature.</span>'
    '<span class="required">8 required</span>',
    unsafe_allow_html=True,
)


# ============================================================
# INPUTS
# ============================================================

col1, col2 = st.columns(2)


with col1:

    x1 = st.number_input(
        "Relative Compactness    X1",
        value=7.0,
        step=1.0,
        format="%.0f",
    )

    x3 = st.number_input(
        "Wall Area    X3",
        value=86.0,
        step=1.0,
        format="%.0f",
    )

    x5 = st.number_input(
        "Overall Height    X5",
        value=354.0,
        step=1.0,
        format="%.0f",
    )

    x7 = st.number_input(
        "Glazing Area    X7",
        value=15.0,
        step=1.0,
        format="%.0f",
    )


with col2:

    x2 = st.number_input(
        "Surface Area    X2",
        value=8.0,
        step=1.0,
        format="%.0f",
    )

    x4 = st.number_input(
        "Roof Area    X4",
        value=65.0,
        step=1.0,
        format="%.0f",
    )

    x6 = st.number_input(
        "Orientation    X6",
        value=24.0,
        step=1.0,
        format="%.0f",
    )

    x8 = st.number_input(
        "Glazing Area Distribution    X8",
        value=87.0,
        step=1.0,
        format="%.0f",
    )


# ============================================================
# BOTTOM AREA
# ============================================================

bottom_col1, bottom_col2 = st.columns([1.8, 1])

with bottom_col1:

    st.markdown(
        '<div class="help-text">'
        'All eight building characteristics are needed<br>'
        'to calculate the heating load.'
        '</div>',
        unsafe_allow_html=True,
    )


with bottom_col2:

    predict = st.button(
        "Predict heating load  →",
        use_container_width=True,
    )


# ============================================================
# PREDICTION
# ============================================================

if predict:

    values = [
        x1,
        x2,
        x3,
        x4,
        x5,
        x6,
        x7,
        x8,
    ]

    try:

        input_data = np.array(values).reshape(1, -1)

        if not np.all(np.isfinite(input_data)):
            raise ValueError

        scaled_data = scaler.transform(input_data)

        prediction = model.predict(scaled_data)[0]

        st.markdown("#### Model output")
        st.metric("Y1 Heating Load", f"{prediction:.2f}")
        st.caption(
            "Predicted heating load based on the eight building characteristics provided."
        )

    except Exception:

        st.error(
            "The prediction could not be generated. "
            "Please check the input values."
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    dedent("""
    <div class="footer-row">
        <span>8 building features / Y1 heating load</span>
        <span>Powered by a trained linear regression model</span>
    </div>
    """),
    unsafe_allow_html=True,
)