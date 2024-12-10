
import gradio as gr
import joblib
import pandas as pd

# Path to the trained model
MODEL_PATH = "tuned_model.pkl"

# Load the model
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}.")

# Define prediction function
def predict_with_model(State: str, Sex: str, GeneralHealth: str, PhysicalHealthDays: str, MentalHealthDays: str, LastCheckupTime: str, PhysicalActivities: str, SleepHours: str, HadStroke: str, HadArthritis: str, HadDiabetes: str, SmokerStatus: str, ECigaretteUsage: str, RaceEthnicityCategory: str, AgeCategory: str, HeightInMeters: str, WeightInKilograms: str, BMI: str, AlcoholDrinkers: str, HighRiskLastYear: str):
    try:
        # Prepare input data as a DataFrame
        input_data = pd.DataFrame([[State, Sex, GeneralHealth, PhysicalHealthDays, MentalHealthDays, LastCheckupTime, PhysicalActivities, SleepHours, HadStroke, HadArthritis, HadDiabetes, SmokerStatus, ECigaretteUsage, RaceEthnicityCategory, AgeCategory, HeightInMeters, WeightInKilograms, BMI, AlcoholDrinkers, HighRiskLastYear]], columns=['State', 'Sex', 'GeneralHealth', 'PhysicalHealthDays', 'MentalHealthDays', 'LastCheckupTime', 'PhysicalActivities', 'SleepHours', 'HadStroke', 'HadArthritis', 'HadDiabetes', 'SmokerStatus', 'ECigaretteUsage', 'RaceEthnicityCategory', 'AgeCategory', 'HeightInMeters', 'WeightInKilograms', 'BMI', 'AlcoholDrinkers', 'HighRiskLastYear'])
        prediction = model.predict(input_data)
        return "Heart Disease Risk" if prediction[0] == 1 else "No Risk"
    except Exception as e:
        return f"Error during prediction: {e}"

# Define the Gradio interface
with gr.Blocks() as app:
    gr.Markdown("# Health Risk Prediction App")
    gr.Markdown("### Input the feature values below to predict health risks.")

    inputs = [
        gr.Dropdown(["Alabama", "Alaska", "Arizona", "Arkansas", "California"], label="State"),  # Example states
        gr.Radio(["Male", "Female"], label="Sex"),
        gr.Radio(["Excellent", "Very Good", "Good", "Fair", "Poor"], label="GeneralHealth"),
        gr.Slider(0, 30, step=1, label="PhysicalHealthDays"),
        gr.Slider(0, 30, step=1, label="MentalHealthDays"),
        gr.Radio(["Within last year", "1-2 years ago", "3-5 years ago", "5+ years ago"], label="LastCheckupTime"),
        gr.Radio(["Yes", "No"], label="PhysicalActivities"),
        gr.Slider(0, 24, step=1, label="SleepHours"),
        gr.Radio(["Yes", "No"], label="HadStroke"),
        gr.Radio(["Yes", "No"], label="HadArthritis"),
        gr.Radio(["Yes", "No"], label="HadDiabetes"),
        gr.Radio(["Smoker", "Non-Smoker"], label="SmokerStatus"),
        gr.Radio(["Yes", "No"], label="ECigaretteUsage"),
        gr.Dropdown(["White", "Black", "Asian", "Hispanic", "Other"], label="RaceEthnicityCategory"),
        gr.Dropdown(["18-24", "25-34", "35-44", "45-54", "55-64", "65+"], label="AgeCategory"),
        gr.Slider(1.0, 2.5, step=0.01, label="HeightInMeters"),
        gr.Slider(30, 200, step=1, label="WeightInKilograms"),
        gr.Slider(10, 50, step=0.1, label="BMI"),
        gr.Radio(["Yes", "No"], label="AlcoholDrinkers"),
        gr.Radio(["Yes", "No"], label="HighRiskLastYear"),
    ]

    predict_button = gr.Button("Predict")
    output = gr.Textbox(label="Prediction Result")

    # Connect prediction logic
    predict_button.click(
        fn=predict_with_model,
        inputs=inputs,
        outputs=output,
    )

# Launch the app
app.launch()
