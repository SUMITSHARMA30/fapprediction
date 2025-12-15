import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="FAP Success Predictor", layout="centered")

# ---------- GLOBAL CSS THEME ----------
st.markdown("""
<style>

/* ===== BACKGROUND (soft lime → teal) ===== */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(
        115deg,
        #e0ffb3 0%,
        #c4f2c3 35%,
        #a5f3fc 75%,
        #67e8f9 100%
    ) !important;
}

/* Center the main content */
.main .block-container {
    max-width: 950px;
    padding: 2.5rem 3rem;
    margin: 0 auto;
    font-family: "Segoe UI", system-ui, sans-serif;
    color: #0f172a;
}

/* ===== HEADERS & TEXT ===== */
h1 {
    font-size: 2.6rem !important;
    font-weight: 800;
    color: #022c22 !important;
    text-align: center;
    margin-bottom: 0.4rem;
}

h2, h3, h4 {
    color: #022c22 !important;
    font-weight: 700;
}

p, label, span {
    color: #0f172a !important;
    font-size: 0.95rem;
}

/* ===== SUCCESS/INFO MESSAGE (model trained) ===== */
.stSuccess {
    background: #dcfce7 !important;
    color: #065f46 !important;
    border-left: 6px solid #22c55e;
    border-radius: 12px;
    font-weight: 600;
}

/* ===== MAIN CARD WRAPPER (.app-card) ===== */
.app-card {
    background: rgba(255, 255, 255, 0.96);
    border-radius: 1.2rem;
    padding: 1.8rem 2rem 2.1rem 2rem;
    margin-top: 1.5rem;
    margin-bottom: 2rem;
    box-shadow: 0 18px 40px rgba(0, 0, 0, 0.18);
    border: 1px solid rgba(148, 163, 184, 0.35);
}

/* ===== METRIC CARDS (top three KPIs) ===== */
[data-testid="stMetric"] {
    background: #ffffff;
    border-radius: 1rem;
    padding: 1rem;
    border: 1px solid #e5e7eb;
    box-shadow: 0 10px 25px rgba(15, 23, 42, 0.08);
}

[data-testid="stMetricLabel"] {
    color: #0f172a !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
}

[data-testid="stMetricValue"] {
    color: #022c22 !important;
    font-weight: 800 !important;
    font-size: 1.4rem !important;
}

/* ===== INPUTS (number/text) ===== */
.stNumberInput input,
.stTextInput input {
    background-color: #ffffff !important;
    color: #0f172a !important;
    border-radius: 0.75rem !important;
    border: 1px solid #cbd5e1 !important;
    padding: 0.45rem 0.75rem !important;
    font-size: 0.95rem !important;
}

/* Input labels */
.stNumberInput label,
.stTextInput label,
.stSelectbox label {
    color: #022c22 !important;
    font-weight: 600 !important;
    margin-bottom: 0.15rem !important;
}

/* ===== SELECTBOX (Station Code & AM) – WHITE ===== */
.stSelectbox > div > div {
    background-color: #ffffff !important;
    color: #0f172a !important;
    border-radius: 0.75rem !important;
    border: 1px solid #cbd5e1 !important;
    min-height: 2.6rem;
}

/* Dropdown arrow */
.stSelectbox svg {
    color: #0f172a !important;
}

/* Dropdown menu container */
[data-baseweb="popover"] {
    background-color: #ffffff !important;
    border-radius: 0.75rem !important;
    border: 1px solid #cbd5e1 !important;
}

/* Dropdown items */
[data-baseweb="menu"] div {
    color: #0f172a !important;
    background-color: #ffffff !important;
}

[data-baseweb="menu"] div:hover {
    background-color: #e2e8f0 !important;
}

/* ===== BUTTON ===== */
.stButton > button {
    background: linear-gradient(90deg, #22c55e, #4ade80);
    color: #052e16 !important;
    border-radius: 999px;
    padding: 0.6rem 2rem;
    font-size: 1rem;
    font-weight: 700;
    border: none;
    box-shadow: 0 12px 30px rgba(34, 197, 94, 0.45);
    transition: all 0.18s ease-in-out;
}

.stButton > button:hover {
    transform: translateY(-1px) scale(1.03);
    box-shadow: 0 20px 40px rgba(34, 197, 94, 0.7);
}

/* ===== WARNING & ERROR (for prediction messages) ===== */
.stWarning {
    background: #fef3c7 !important;
    color: #92400e !important;
    border-left: 6px solid #facc15;
    border-radius: 12px;
    font-weight: 600;
}

.stError {
    background: #fee2e2 !important;
    color: #991b1b !important;
    border-left: 6px solid #ef4444;
    border-radius: 12px;
    font-weight: 600;
}

/* ===== SCROLLBAR ===== */
::-webkit-scrollbar {
    width: 8px;
}
::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #22c55e, #4ade80);
    border-radius: 999px;
}

/* ===== HIDE STREAMLIT BRANDING ===== */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

</style>
""", unsafe_allow_html=True)



# ================================
# 1. PAGE CONFIG
# ================================
st.set_page_config(page_title="FAP Success Predictor", layout="centered")
st.title("📦 FAP Success Percentage Predictor")
st.write("Predict FAP Success % using pickup operations data.")

# ================================
# 2. LOAD & PREPARE DATA
# ================================

@st.cache_data
def load_and_train():
    df = pd.read_csv(r"C:\Users\Sumit\Downloads\project.csv")
    df.columns = df.columns.str.strip()

    # keep only rows with target
    df = df.dropna(subset=["FAP Success Per"])

    # drop fully empty columns and RCA
    df = df.dropna(axis=1, how="all")
    df = df.drop(columns=["RCA"], errors="ignore")

    # numeric conversion
    num_cols = [
        "Total Pickup Assigned", "Pickup Done", "FAP",
        "FAP Success", "Pickup Canceled", "QC Failure",
        "Pickup Failure", "Total Pickup Per", "FAP Success Per"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # simple missing handling
    for c in df.select_dtypes(include=["int64", "float64"]).columns:
        df[c] = df[c].fillna(df[c].median())

    for c in df.select_dtypes(include="object").columns:
        if df[c].notna().sum() > 0:
            df[c] = df[c].fillna(df[c].mode().iloc[0])
        else:
            df[c] = df[c].fillna("Unknown")



    # build regression data
    TARGET = "FAP Success Per"
    X = df.drop(columns=[TARGET])
    X = pd.get_dummies(X, drop_first=True)
    X = X.fillna(0)
    y = df[TARGET]

    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # scaler + model
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    model = LinearRegression()
    model.fit(X_train_s, y_train)

    return df, X, y, model, scaler

df, X_all, y_all, model, scaler = load_and_train()

st.success("Model trained on historical data.")

# ================================
# 3. INPUT FORM FOR PREDICTION
# ================================
st.subheader("🔮 Predict FAP Success % for a new scenario")
colA, colB, colC = st.columns(3)
colA.metric("Train Samples", len(X_all))
colB.metric("Features Used", X_all.shape[1])
colC.metric("Target", "FAP Success %")

# We will take only core numeric + few categorical inputs
col1, col2 = st.columns(2)

with col1:
    total_assigned = st.number_input(
        "Total Pickup Assigned", min_value=0, max_value=500, value=10
    )
    pickup_done = st.number_input(
        "Pickup Done", min_value=0, max_value=500, value=5
    )
    fap = st.number_input(
        "FAP (First Attempt Pickups)", min_value=0, max_value=500, value=5
    )
    fap_success = st.number_input(
        "FAP Success", min_value=0, max_value=500, value=3
    )

with col2:
    pickup_canceled = st.number_input(
        "Pickup Canceled", min_value=0, max_value=500, value=0
    )
    qc_failure = st.number_input(
        "QC Failure", min_value=0, max_value=500, value=0
    )
    pickup_failure = st.number_input(
        "Pickup Failure", min_value=0, max_value=500, value=1
    )
    total_pickup_per = st.number_input(
        "Total Pickup % (Total Pickup Per)", min_value=0.0, max_value=100.0, value=20.0
    )

# categorical: station_code, AM (from existing unique values)
station_codes = sorted(df["station_code"].unique().tolist()) if "station_code" in df.columns else []
ams = sorted(df["AM"].unique().tolist()) if "AM" in df.columns else []

if station_codes:
    station_code = st.selectbox("Station Code", station_codes)
else:
    station_code = None

if ams:
    am = st.selectbox("AM (Area Manager)", ams)
else:
    am = None

# ================================
# 4. BUILD A SINGLE INPUT ROW
# ================================
if st.button("Predict FAP Success %"):
    # start from a base row (use first row of df as template)
    base = df.iloc[[0]].copy()

    # overwrite numeric features
    if "Total Pickup Assigned" in base.columns:
        base["Total Pickup Assigned"] = total_assigned
    if "Pickup Done" in base.columns:
        base["Pickup Done"] = pickup_done
    if "FAP" in base.columns:
        base["FAP"] = fap
    if "FAP Success" in base.columns:
        base["FAP Success"] = fap_success
    if "Pickup Canceled" in base.columns:
        base["Pickup Canceled"] = pickup_canceled
    if "QC Failure" in base.columns:
        base["QC Failure"] = qc_failure
    if "Pickup Failure" in base.columns:
        base["Pickup Failure"] = pickup_failure
    if "Total Pickup Per" in base.columns:
        base["Total Pickup Per"] = total_pickup_per

    # overwrite categorical features
    if station_code is not None and "station_code" in base.columns:
        base["station_code"] = station_code
    if am is not None and "AM" in base.columns:
        base["AM"] = am

    # drop target if present
    if "FAP Success Per" in base.columns:
        base = base.drop(columns=["FAP Success Per"])

    # match training preprocessing
    base_dummies = pd.get_dummies(base, drop_first=True)
    base_dummies = base_dummies.reindex(columns=X_all.columns, fill_value=0)

    # scale and predict
    base_scaled = scaler.transform(base_dummies)
    pred = model.predict(base_scaled)[0]

    st.subheader("🧾 Prediction Result")
    st.write(f"**Predicted FAP Success Per:** `{pred:.2f} %`")

    if pred < 20:
        st.error("Low predicted FAP success. Consider improving operations.")
    elif pred < 40:
        st.warning("Average performance. Some optimization needed.")
    else:
        st.success("Good predicted FAP performance. Keep it up! 💪")
