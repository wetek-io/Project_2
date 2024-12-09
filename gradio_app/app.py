"""
Webapp Front End
"""

import gradio as gr
import pandas as pd
import joblib
from data.clean_data import fetch_check

from data.value_maps import category_maps, binary_maps

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

binary_inputs = {
    feature: gr.Radio(
        choices=list(mapping.keys()),
        label=feature.replace("_", " "),
    )
    for feature, mapping in binary_maps.items()
    if mapping
}

# Generate dropdowns/radio buttons for categorical features using value_maps
categorical_inputs = {
    feature: gr.Dropdown(
        choices=list(mapping.keys()),
        label=feature.replace("_", " "),
    )
    for feature, mapping in category_maps.items()
    if mapping  # Only include features with mappings
}

input_types = list(categorical_inputs.values()) + list(binary_inputs.values())


def predict_outcome(*user_inputs):
    """
    Converts user inputs into model-friendly format, runs the prediction,
    and returns the result.
    """
    # List of features expected by the model
    expected_features = rf_model.feature_names_in_  # Use this if your model supports it
    input_data = dict(zip(expected_features, user_inputs))

    # Ensure all required features are present
    for feature in expected_features:
        if feature not in input_data:
            input_data[feature] = 0  # Default value for missing features

    # Create a DataFrame for prediction
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
    outputs = gr.Label(label="Prediction")
    return gr.Interface(fn=predict_outcome, inputs=input_types, outputs=outputs)


# Run the app
if __name__ == "__main__":
    interface = build_interface()
    interface.launch()
