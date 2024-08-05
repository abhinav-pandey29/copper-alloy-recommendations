import streamlit as st
import pandas as pd

from composition_predictor import run_inverse_model

st.title("Copper Alloy Recommendations")
st.text(
    "Get recommendations of copper alloy compositions based "
    "on required Thermal conductivity."
)

value = st.number_input(label="Thermal Conductivity", key="value-input")

if st.button("Get recommendations"):

    with st.spinner("Generating predictions..."):
        # TODO: Fix model loading issue.
        pred_df = run_inverse_model("thermal_conductivity", value, n=10)
        zero_cols = pred_df.columns[(pred_df == 0).all()]
        pred_df.drop(columns=zero_cols, inplace=True)

    st.table(pred_df)
