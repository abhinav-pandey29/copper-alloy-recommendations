import pandas as pd
import streamlit as st

from composition_predictor import run_inverse_model

# Set the page configuration
st.set_page_config(
    page_title="Copper Alloy Recommendations",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar for input
st.sidebar.title("Copper Alloy Recommendations")
st.sidebar.write(
    "Get recommendations of copper alloy compositions based on required thermal conductivity."
)

# Thermal Conductivity input
thermal_conductivity = st.sidebar.number_input(
    label="Enter Desired Thermal Conductivity (BTU·h⁻¹·ft⁻¹·℉⁻¹)",
    min_value=0.0,
    max_value=1000.0,
    value=220.0,
    step=0.1,
    format="%.1f",
    key="thermal_conductivity_input",
)


# Main page content
st.title("Copper Alloy Recommendations")
st.write(
    """
    This app provides recommendations for copper alloy compositions based on the desired thermal conductivity.
    Simply enter the required thermal conductivity in the sidebar, and click "Get Recommendations" to see the 
    suggested compositions.

    If any known alloys from [Copper.org](https://alloys.copper.org/) match the specified thermal conductivity
    value, they will be included in the recommendations and marked with a 100% confidence level.
    """
)


# Button to get recommendations
if st.sidebar.button("Get Recommendations"):

    with st.spinner("Generating predictions..."):
        try:
            # Generate predictions
            pred_df = run_inverse_model(
                "thermal_conductivity", thermal_conductivity, n=300
            )

            # Drop columns that are entirely zero
            zero_cols = pred_df.columns[(pred_df == 0).all()]
            pred_df.drop(columns=zero_cols, inplace=True)

            st.success("Recommendations generated successfully!")

            # Display the recommendations
            st.write(
                f"### Recommended Copper Alloy Compositions for Thermal Conductivity of {thermal_conductivity} (BTU·h⁻¹·ft⁻¹·℉⁻¹)"
            )
            # TODO: Simplify result formatting steps
            st.table(
                pred_df.round(3)
                .replace(0, None)
                .drop_duplicates()
                .reset_index(drop=True)
                .head(10)
                .astype(str)
                .replace("None", None)
                .style.highlight_null(props="color: transparent;")
            )
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Author reference and methodology
st.sidebar.write(
    """
    ---
    ### 📄 Methodology
    *Recommendations are based on the methodology from my final year thesis. Read the full project report [here](https://github.com/abhinav-pandey29/Projects/blob/main/Copper%20Alloy%20Discovery%20using%20AI/project/Project%20Report.pdf).*
    
    ---
    ### 👨‍💻 Author
    > **Abhinav Pandey**  
    > Bachelor's of Advanced Computing (Honours)  
    > Australian National University  
    > 2020
    """
)
