"""
Webapp Front End
"""

import gradio as gr
import joblib
from data.clean_data import fetch_check

from data.value_maps import category_maps, binary_maps

MODEL_PATH = "Random_Foresttest_model.pkl"
DEFAULT_VALUE = 99
try:
    rf_model = joblib.load(MODEL_PATH)
    joblib.dump(rf_model, MODEL_PATH)
except FileNotFoundError as e:
    raise FileNotFoundError(
        f"Model file not found at {MODEL_PATH}. Please check the path."
    ) from e

og_df = fetch_check(to_fetch=True, to_fillna=True, to_dropna=True)

binary_inputs = {
    feature: gr.Radio(
        choices=list(mapping.keys()),
        label=feature.replace("_", " "),
    )
    for feature, mapping in binary_maps.items()
    if mapping
}

categorical_inputs = {
    feature: gr.Dropdown(
        choices=list(mapping.keys()),
        label=feature.replace("_", " "),
    )
    for feature, mapping in category_maps.items()
    if mapping
}

input_types = list(categorical_inputs.values()) + list(binary_inputs.values())

for i in categorical_inputs:
    print(f"input_types: {i}")
for i in binary_inputs:
    print(f"input_types: {i}")
for i in input_types:
    print(f"input_types: {i}")


def predict_outcome(*user_inputs):
    """
    Converts user inputs into model-friendly format, runs the prediction,
    and returns the result.
    """
    # Use maps to set expected features
    expected_features = list(categorical_inputs.keys()) + list(binary_inputs.keys())

    input_data = dict(zip(expected_features, user_inputs))

    # Ensure all required features are present and that the numerical values are used for the model
    input_data = {}
    for feature, user_input in zip(expected_features, user_inputs):
        if feature in binary_maps:
            # Convert 'Yes'/'No' to 1/0
            input_data[feature] = binary_maps[feature].get(user_input, DEFAULT_VALUE)
        elif feature in category_maps:
            # Convert categorical values
            input_data[feature] = category_maps[feature].get(user_input, DEFAULT_VALUE)
        else:
            # Default value for unexpected inputs
            input_data[feature] = DEFAULT_VALUE

    # Create a DataFrame for prediction
    input_df = pd.DataFrame([input_data])[expected_features]

    # Perform prediction
    try:
        prediction = rf_model.predict(input_df)[0]
        return "High Risk" if prediction == 1 else "Low Risk"
    except ValueError as e:
        raise ValueError(f"Error during prediction: {e}") from e


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
