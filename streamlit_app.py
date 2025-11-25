
import io
import pathlib
import pandas as pd
import joblib
import streamlit as st

MODEL_PATH = pathlib.Path("model_filtered.joblib")

REQUIRED_COLUMNS = [
    "year", "item", "district",
    "prectotcorr_ann", "prectotcorr_sum",
    "qv2m_ann", "rh2m_ann", "t2m_max_ann", "t2m_range_ann",
]

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    bundle = joblib.load(MODEL_PATH)
    # Expecting a dict: {"pipe": <sklearn-pipeline>, "num_feats": [...], "cat_feats": [...]}
    return bundle["pipe"], bundle["num_feats"], bundle["cat_feats"]

pipe, num_feats, cat_feats = load_model()

st.set_page_config(page_title="Livestock Viability (LSU/ha)", layout="wide")
st.title("Livestock Viability — LSU/ha model")
st.caption("Trained on ≤2017, predicting ≥2018 using climate + species + district")

with st.sidebar:
    st.header("Inputs required")
    st.markdown(
        """
        - `year` (≥ 2018)
        - `item` (Cattle / Sheep / Goats)
        - `district` (e.g., KSD)
        - `prectotcorr_ann`, `prectotcorr_sum`
        - `qv2m_ann`, `rh2m_ann`, `t2m_max_ann`, `t2m_range_ann`
        """
    )

st.subheader("Single-row prediction")

with st.form("single_row"):
    c1, c2, c3 = st.columns(3)
    with c1:
        year = st.number_input("year", min_value=2018, max_value=2100, value=2025, step=1)
        item = st.selectbox("item", ["Cattle", "Sheep", "Goats"])
        district = st.text_input("district", value="South Africa")
    with c2:
        prectotcorr_ann = st.number_input("prectotcorr_ann", value=1.2)
        prectotcorr_sum = st.number_input("prectotcorr_sum", value=450.0)
        qv2m_ann = st.number_input("qv2m_ann", value=6.8)
    with c3:
        rh2m_ann = st.number_input("rh2m_ann", value=58.0)
        t2m_max_ann = st.number_input("t2m_max_ann", value=40.0)
        t2m_range_ann = st.number_input("t2m_range_ann", value=15.0)

    submitted = st.form_submit_button("Predict")

if submitted:
    # Build a single-row DataFrame and align columns to the trained pipeline
    row = pd.DataFrame([{
        "year": int(year),
        "item": item,
        "district": district,
        "prectotcorr_ann": float(prectotcorr_ann),
        "prectotcorr_sum": float(prectotcorr_sum),
        "qv2m_ann": float(qv2m_ann),
        "rh2m_ann": float(rh2m_ann),
        "t2m_max_ann": float(t2m_max_ann),
        "t2m_range_ann": float(t2m_range_ann),
    }])

    # Order/features expected by the pipeline
    expected_cols = num_feats + cat_feats
    try:
        # Reindex to ensure exact order; will raise if a column is missing
        row = row.reindex(columns=expected_cols)
    except Exception as e:
        st.error(f"Column alignment failed. Check your saved feature lists. Details: {e}")
    else:
        try:
            pred = pipe.predict(row)[0]
            st.success(f"Predicted LSU/ha: {pred:.3f}")
        except Exception as e:
            st.error(f"Prediction failed. Details: {e}")

