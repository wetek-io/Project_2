import gradio as gr
import joblib
import pandas as pd

# Load the pre-trained model
model = joblib.load("tuned_model.pkl")

# Load the features used during training
features = pd.read_csv("features_used_in_model.csv")["Feature"].tolist()

# Prediction function
def predict_heart_failure(*input_values):
    try:
        # Convert inputs into a dictionary
        input_data = dict(zip(features, input_values))
        
        # Convert input dictionary to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Predict probability for heart failure (class 1)
        probability = model.predict_proba(input_df)[:, 1][0]
        
        # Predict class (0 or 1)
        prediction = "At Risk of Heart Failure" if probability >= 0.3 else "No Risk Detected"
        
        # Return prediction, probability, and user inputs
        return prediction, round(probability, 4), input_data
    except Exception as e:
        return "Error", 0, {"error": str(e)}

# Gradio Interface
inputs = [gr.Textbox(label=feature, placeholder=f"Enter value for {feature}") for feature in features]

interface = gr.Interface(
    fn=predict_heart_failure,
    inputs=inputs,
    outputs=[
        gr.Text(label="Prediction"),
        gr.Number(label="Risk Probability"),
        gr.JSON(label="User Inputs")
    ],
    title="Heart Failure Prediction Model",
    description=(
        "Predicts the likelihood of heart failure based on health features. "
        "Enter the values for the features below and receive the prediction."
    )
)

# Launch the interface for local testing or Hugging Face Spaces deployment
if __name__ == "__main__":
    interface.launch(share=True)