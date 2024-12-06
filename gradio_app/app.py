import gradio as gr
import pandas as pd
import joblib

# Load the trained model
model_path = "trained_model.pkl"
rf_model = joblib.load(model_path)

# Define feature ranges and labels based on data
numerical_features = ['BMI', 'WeightInKilograms', 'HeightInMeters', 'PhysicalHealthDays', 'SleepHours']
categorical_features = [
    'HadAngina_Yes', 'HadHeartAttack_Yes', 'ChestScan_Yes', 
    'HadStroke_Yes', 'DifficultyWalking_Yes', 'HadDiabetes_Yes', 
    'PneumoVaxEver_Yes', 'HadArthritis_Yes'
]

# Define sliders for numerical features
sliders = {
    "BMI": (0, 50, 1),
    "WeightInKilograms": (30, 200, 1),
    "HeightInMeters": (1.0, 2.5, 0.01),
    "PhysicalHealthDays": (0, 30, 1),
    "SleepHours": (0, 24, 1)
}

# Define radio buttons for categorical features
radio_options = ['Yes', 'No']

# Prediction function
def predict_outcome(*inputs):
    input_data = dict(zip(numerical_features + categorical_features, inputs))
    
    # Convert categorical inputs to numerical
    for feature in categorical_features:
        input_data[feature] = 1 if input_data[feature] == "Yes" else 0
    
    # Create input DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Predict using the model
    prediction = rf_model.predict(input_df)[0]
    prediction_label = "High Risk" if prediction == 1 else "Low Risk"
    
    # Display input values for debugging
    return prediction_label, input_data

# Build Gradio interface
inputs = [
    gr.Slider(sliders[feature][0], sliders[feature][1], sliders[feature][2], label=feature) 
    for feature in numerical_features
] + [
    gr.Radio(radio_options, label=feature) for feature in categorical_features
]

outputs = [
    gr.Textbox(label="Prediction"),
    gr.JSON(label="Input Values (Debugging)")
]

interface = gr.Interface(
    fn=predict_outcome,
    inputs=inputs,
    outputs=outputs,
    title="Health Risk Prediction with Debugging",
    description="Predicts health risks based on input parameters using the trained model. Includes input values for debugging."
)

# Launch the app
if __name__ == "__main__":
    interface.launch()