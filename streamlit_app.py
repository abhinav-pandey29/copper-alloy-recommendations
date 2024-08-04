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
        pred_df = run_inverse_model("Thermal Conductivity", value, n=10)

    st.table(pred_df)
