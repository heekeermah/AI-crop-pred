
import streamlit as st
import pandas as pd
import joblib


model = joblib.load("crop_prediction_model.pkl")


st.set_page_config(
    page_title="Crop Advisor AI",
    page_icon="🌱"
)


st.title("🌱 Crop Advisor AI")
st.subheader("Extension Agent Decision Support Tool")

st.write(
    """
This tool helps agricultural extension agents recommend suitable crops
using measured soil and environmental conditions.
"""
)


N = st.number_input("Nitrogen (N)", min_value=0, max_value=120, value=0, step=1)
P = st.number_input("Phosphorus (P)", min_value=5, max_value=80, value=5, step=1)
K = st.number_input("Potassium (K)", min_value=5, max_value=85, value=5, step=1)

temperature = st.number_input(
    "Temperature (°C)",
    min_value=10.0, max_value=35.00, value=10.00, step=0.01 
)

humidity = st.number_input(
    "Humidity (%)",
   min_value=14.0, max_value=95.00, value=14.00, step=0.01
)

ph = st.number_input(
    "Soil pH",
    min_value=5.0, max_value=9.00, value=5.00, step=0.01
)

rainfall = st.number_input(
    "Rainfall (mm)",
    min_value=20.0, max_value=300.00, value=20.00, step=0.01
)

soil = st.selectbox(
    "Soil Texture",
    [
        "loamy",
        "sandy loam",
        "loamy clay",
        "clay",
        "sandy"
    ]
)


if st.button("Generate Crop Recommendation"):

    input_data = pd.DataFrame({
        "N":[N],
        "P":[P],
        "K":[K],
        "temperature":[temperature],
        "humidity":[humidity],
        "ph":[ph],
        "rainfall":[rainfall],
        "soil_texture":[soil]
    })


    probabilities = model.predict_proba(input_data)[0]

    crops = model.classes_

    results = (
        pd.DataFrame({
            "Crop": crops,
            "Probability":probabilities
        })
        .sort_values(
            "Probability",
            ascending=False
        )
        .head(3)
    )


    st.success(
        f"Primary Recommendation: {results.iloc[0]['Crop'].title()}"
    )


    st.write("### Alternative Suitable Crops")

    for _, row in results.iterrows():

        st.write(
            f"- {row['Crop'].title()} "
            f"({row['Probability']*100:.1f}% suitability)"
        )


    crop = results.iloc[0]["Crop"]


    st.write("### Why this crop is suitable")

    st.write(
        f"""
The recommendation for **{crop.title()}** is based on the combined
relationship between:

- available soil nutrients (Nitrogen, Phosphorus and Potassium);
- soil texture ({soil});
- temperature conditions ({temperature}°C);
- humidity level ({humidity}%);
- rainfall availability ({rainfall} mm);
- soil acidity/alkalinity (pH {ph}).

The model identified this crop as the most compatible option
based on historical crop and soil patterns in the dataset.
"""
    )
