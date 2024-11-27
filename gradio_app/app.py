
import gradio as gr
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("optimized_heart_failure_model.pkl")

# Define the prediction function
def predict_heart_failure(**inputs):
    # Encode inputs as DataFrame (match the reduced feature set)
    input_data = pd.DataFrame(inputs, index=[0])
    # Make prediction
    prediction = model.predict(input_data)[0]
    return "High likelihood of heart failure" if prediction == 1 else "Low likelihood of heart failure"

# Get input features from the reduced model
feature_columns = [
    "PhysicalHealthDays", "MentalHealthDays", "SleepHours", "BMI",
    "Sex_Male", "GeneralHealth_Good", "GeneralHealth_Very good",
    "GeneralHealth_Excellent"
]

# Generate Gradio inputs dynamically based on features
gradio_inputs = []
for feature in feature_columns:
    if feature.startswith("Sex_"):
        gradio_inputs.append(gr.Radio(["Male", "Female"], label="Sex"))
    elif feature.startswith("GeneralHealth_"):
        gradio_inputs.append(gr.Radio(["Poor", "Fair", "Good", "Very good", "Excellent"], label="General Health"))
    elif feature == "PhysicalHealthDays":
        gradio_inputs.append(gr.Slider(0, 30, step=1, label="Physical Health Days"))
    elif feature == "MentalHealthDays":
        gradio_inputs.append(gr.Slider(0, 30, step=1, label="Mental Health Days"))
    elif feature == "SleepHours":
        gradio_inputs.append(gr.Slider(0, 12, step=0.5, label="Sleep Hours"))
    elif feature == "BMI":
        gradio_inputs.append(gr.Slider(10, 50, step=0.1, label="BMI"))

# Map Gradio inputs to a dictionary for the predict function
def input_to_dict(*args):
    feature_dict = {feature: val for feature, val in zip(feature_columns, args)}
    # Encode categorical inputs into binary format (e.g., Male -> 1, Female -> 0)
    feature_dict["Sex_Male"] = 1 if feature_dict.get("Sex_Male", "Male") == "Male" else 0
    general_health = feature_dict.get("GeneralHealth_Good", "Poor")
    for category in ["Good", "Very good", "Excellent"]:
        feature_dict[f"GeneralHealth_{category}"] = 1 if general_health == category else 0
    return feature_dict

# Set up Gradio interface
interface = gr.Interface(
    fn=lambda *args: predict_heart_failure(**input_to_dict(*args)),
    inputs=gradio_inputs,
    outputs="text",
    title="Heart Failure Risk Prediction (Optimized)"
)

# Launch the app
if __name__ == "__main__":
    interface.launch()
