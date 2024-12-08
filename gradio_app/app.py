"""
Webapp Front End
"""

import gradio as gr
import pandas as pd
import joblib
from data.clean_data import fetch_check

from data.value_maps import value_maps

# Load the trained model
MODEL_PATH = "gradio_app/trained_model.pkl"
try:
    rf_model = joblib.load(MODEL_PATH)
except FileNotFoundError as e:
    raise FileNotFoundError(
        f"Model file not found at {MODEL_PATH}. Please check the path."
    ) from e

# Fetch and clean original data
og_df = fetch_check(to_fetch=True, to_fillna=True, to_dropna=True)


def compute_slider_limits(df, threshold=99):
    """
    Dynamically create and compute sliders
    """
    return {
        col: {
            "min": int(df[col].min()),
            "max": int(df[df[col] < threshold][col].max()),
        }
        for col in df.select_dtypes(include=["float64", "int64"]).columns
    }


# Set sliders
sliders = compute_slider_limits(og_df, threshold=99)

# Define radio buttons for categorical features
radio_options = ["Yes", "No"]


def predict_outcome(*inputs):
    """
    Converts user inputs into model-friendly format, runs the prediction,
    and returns the result.
    """
    # Convert inputs to a DataFrame for prediction
    input_data = dict(zip(sliders.keys(), inputs))

    # Map categorical inputs to numerical
    for feature, value in input_data.items():
        if value in radio_options:
            input_data[feature] = 1 if value == "Yes" else 0

    input_df = pd.DataFrame([input_data])

    # Perform prediction
    try:
        prediction = rf_model.predict(input_df)[0]
        prediction_label = "High Risk" if prediction == 1 else "Low Risk"
    except ValueError as e:
        raise ValueError(f"Error during prediction: {e}") from e

    return prediction_label


def build_interface():
    """
    Constructs the Gradio interface dynamically based on the dataset.
    """
    # inputs = [
    #     gr.Slider(
    #         minimum=sliders[col]["min"],
    #         maximum=sliders[col]["max"],
    #         label=col,
    #     )
    #     for col in sliders
    # ]
    inputs = [
        gr.Dropdown(
            choices=list(mapping.keys()),  # Use human-readable values
            label=feature.replace("_", " "),  # Clean up labels for display
        )
        for feature, mapping in value_maps.items()
    ]
    # Add radio buttons for categorical features if needed
    inputs += [
        gr.Radio(choices=radio_options, label=col) for col in []
    ]  # Add categorical if applicable

    outputs = gr.Label(label="Prediction")
    return gr.Interface(fn=predict_outcome, inputs=inputs, outputs=outputs)


# Run the app
if __name__ == "__main__":
    interface = build_interface()
    interface.launch()
