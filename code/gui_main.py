import customtkinter as ctk
import joblib
from code.model import train_and_save_model
from code.visualizeData import visualize_heart_rate, visualize_bmi, visualize_blood_sugar

# Function to load a trained model
def load_model(file_name):
    try:
        return joblib.load(file_name)
    except FileNotFoundError:
        ctk.CTkMessagebox.show_error("Model Error", f"Model file '{file_name}' not found.")
        return None

# Function to normalize user-input heart rate
def normalize_input_heart_rate(value):
    min_heart_rate = 60
    max_heart_rate = 160
    if value < min_heart_rate or value > max_heart_rate:
        raise ValueError(f"Heart rate must be between {min_heart_rate} and {max_heart_rate}.")
    return (value - min_heart_rate) / (max_heart_rate - min_heart_rate)

# Predict health risk
def predict_health_risk():
    def get_prediction():
        try:
            heart_rate = float(heart_rate_entry.get())
            activity_level = float(activity_level_entry.get())

            # Normalize heart rate
            heart_rate_scaled = normalize_input_heart_rate(heart_rate)

            # Validate activity level
            if not (0 <= activity_level <= 1):
                ctk.CTkMessagebox.show_error("Input Error", "Activity level must be between 0 and 1.")
                return

            # Load the model
            model = load_model('health_model.pkl')
            if model:
                prediction = model.predict([[heart_rate_scaled, activity_level]])[0]
                result_label.configure(
                    text=f"Predicted Risk Level: {prediction.upper()}",
                    font=("Helvetica", 16, "bold"),
                    text_color=("red" if prediction.upper() == "HIGH" else
                                "orange" if prediction.upper() == "MEDIUM" else "green")
                )
            else:
                result_label.configure(text="Error: Model not found.", font=("Helvetica", 16), text_color="red")
        except ValueError as e:
            ctk.CTkMessagebox.show_error("Input Error", str(e))

    # Health Risk GUI window
    health_risk_window = ctk.CTkToplevel(root)
    health_risk_window.title("Predict Health Risk")
    health_risk_window.geometry("400x400")

    ctk.CTkLabel(health_risk_window, text="Enter Heart Rate (60-160):", font=("Helvetica", 14)).pack(pady=10)
    heart_rate_entry = ctk.CTkEntry(health_risk_window, width=300)
    heart_rate_entry.pack(pady=5)

    ctk.CTkLabel(health_risk_window, text="Enter Activity Level (0-1):", font=("Helvetica", 14)).pack(pady=10)
    activity_level_entry = ctk.CTkEntry(health_risk_window, width=300)
    activity_level_entry.pack(pady=5)

    ctk.CTkButton(health_risk_window, text="Predict", command=get_prediction, width=150).pack(pady=20)

    result_label = ctk.CTkLabel(health_risk_window, text="", font=("Helvetica", 16))
    result_label.pack(pady=10)

# BMI Calculation
def calculate_bmi():
    def get_bmi_prediction():
        try:
            height = float(height_entry.get())
            weight = float(weight_entry.get())

            if height <= 0 or weight <= 0:
                ctk.CTkMessagebox.show_error("Input Error", "Height and weight must be positive values.")
                return

            bmi = round(weight / (height ** 2), 1)  # Calculate BMI

            bmi_model = load_model('bmi_model.pkl')  # Load the BMI model
            if bmi_model:
                category = bmi_model.predict([[height, weight]])[0]
                result_label.configure(
                    text=f"BMI: {bmi} ({category})",
                    font=("Helvetica", 16, "bold"),
                    text_color="green" if category == "Normal" else "red"
                )
            else:
                result_label.configure(
                    text="Error: Model not loaded.",
                    font=("Helvetica", 16),
                    text_color="red"
                )
        except ValueError:
            ctk.CTkMessagebox.show_error("Input Error", "Please enter valid numeric values for height and weight.")

    # BMI GUI window
    bmi_window = ctk.CTkToplevel(root)
    bmi_window.title("Calculate BMI")
    bmi_window.geometry("400x400")

    ctk.CTkLabel(bmi_window, text="Enter Height (m):", font=("Helvetica", 14)).pack(pady=10)
    height_entry = ctk.CTkEntry(bmi_window, width=300)
    height_entry.pack(pady=5)

    ctk.CTkLabel(bmi_window, text="Enter Weight (kg):", font=("Helvetica", 14)).pack(pady=10)
    weight_entry = ctk.CTkEntry(bmi_window, width=300)
    weight_entry.pack(pady=5)

    ctk.CTkButton(bmi_window, text="Calculate", command=get_bmi_prediction, width=150).pack(pady=20)

    result_label = ctk.CTkLabel(bmi_window, text="", font=("Helvetica", 16))
    result_label.pack(pady=10)

# Blood Sugar Prediction
def predict_blood_sugar():
    def get_sugar_prediction():
        try:
            fasting_sugar = float(sugar_entry.get())

            sugar_model = load_model('blood_sugar_model.pkl')
            if sugar_model:
                category = sugar_model.predict([[fasting_sugar]])[0]
                result_label.configure(
                    text=f"Blood Sugar: {category}",
                    font=("Helvetica", 16, "bold"),
                    text_color="green" if category == "Normal" else "red"
                )
            else:
                result_label.configure(
                    text="Error: Model not loaded.",
                    font=("Helvetica", 16),
                    text_color="red"
                )
        except ValueError:
            ctk.CTkMessagebox.show_error("Input Error", "Please enter a valid fasting blood sugar value.")

    # Blood Sugar GUI window
    sugar_window = ctk.CTkToplevel(root)
    sugar_window.title("Predict Blood Sugar")
    sugar_window.geometry("400x400")

    ctk.CTkLabel(sugar_window, text="Enter Fasting Blood Sugar (mg/dL):", font=("Helvetica", 14)).pack(pady=10)
    sugar_entry = ctk.CTkEntry(sugar_window, width=300)
    sugar_entry.pack(pady=5)

    ctk.CTkButton(sugar_window, text="Predict", command=get_sugar_prediction, width=150).pack(pady=20)

    result_label = ctk.CTkLabel(sugar_window, text="", font=("Helvetica", 16))
    result_label.pack(pady=10)

# GUI setup
ctk.set_appearance_mode("Dark")  # Options: "System", "Light", "Dark"
ctk.set_default_color_theme("dark-blue")  # Options: "blue", "green", "dark-blue"

root = ctk.CTk()
root.title("AI-Powered Health Monitoring System")
root.geometry("800x600")
root.resizable(True, True)

# Header
header_frame = ctk.CTkFrame(root, height=100, corner_radius=0)
header_frame.pack(fill="x", side="top")

header_label = ctk.CTkLabel(
    header_frame,
    text="AI-Powered Health Monitoring System",
    font=("Helvetica", 24, "bold"),
    text_color="white",
)
header_label.pack(pady=20)

# Content
content_frame = ctk.CTkFrame(root)
content_frame.pack(fill="both", expand=True, padx=20, pady=20)

# Tabs
tabview = ctk.CTkTabview(content_frame, width=700)
tabview.pack(fill="both", expand=True)

tabview.add("Home")
tabview.add("Training")
tabview.add("Prediction")
tabview.add("Visualization")

# Home Tab
home_tab = tabview.tab("Home")
home_label = ctk.CTkLabel(
    home_tab, text="Welcome to the Health Monitoring System", font=("Helvetica", 18)
)
home_label.pack(pady=30)
home_description = ctk.CTkLabel(
    home_tab,
    text="Monitor and predict health parameters like heart rate, BMI, and blood sugar "
         "levels using AI-powered models.",
    font=("Helvetica", 14),
    text_color="lightgray",
)
home_description.pack(pady=10)

# Training Tab
training_tab = tabview.tab("Training")
train_button = ctk.CTkButton(
    training_tab,
    text="Train All Models",
    font=("Helvetica", 16),
    width=300,
    command=train_and_save_model,
)
train_button.pack(pady=30)
training_label = ctk.CTkLabel(
    training_tab,
    text="Click the button to train models for predicting health risks, BMI, and blood sugar levels.",
    font=("Helvetica", 14),
    text_color="lightgray",
)
training_label.pack(pady=10)

# Prediction Tab
prediction_tab = tabview.tab("Prediction")

health_risk_button = ctk.CTkButton(
    prediction_tab,
    text="Predict Health Risk",
    font=("Helvetica", 16),
    width=300,
    command=predict_health_risk
)
health_risk_button.pack(pady=20)

bmi_button = ctk.CTkButton(
    prediction_tab,
    text="Calculate BMI",
    font=("Helvetica", 16),
    width=300,
    command=calculate_bmi
)
bmi_button.pack(pady=20)

blood_sugar_button = ctk.CTkButton(
    prediction_tab,
    text="Predict Blood Sugar",
    font=("Helvetica", 16),
    width=300,
    command=predict_blood_sugar
)
blood_sugar_button.pack(pady=20)

# Visualization Tab
visualization_tab = tabview.tab("Visualization")

visualize_heart_rate_button = ctk.CTkButton(
    visualization_tab,
    text="Visualize Heart Rate",
    font=("Helvetica", 16),
    width=300,
    command=visualize_heart_rate,
)
visualize_heart_rate_button.pack(pady=15)

visualize_bmi_button = ctk.CTkButton(
    visualization_tab,
    text="Visualize BMI",
    font=("Helvetica", 16),
    width=300,
    command=visualize_bmi,
)
visualize_bmi_button.pack(pady=15)

visualize_blood_sugar_button = ctk.CTkButton(
    visualization_tab,
    text="Visualize Blood Sugar",
    font=("Helvetica", 16),
    width=300,
    command=visualize_blood_sugar,
)
visualize_blood_sugar_button.pack(pady=15)

# Footer
footer_frame = ctk.CTkFrame(root, height=50, corner_radius=0)
footer_frame.pack(fill="x", side="bottom")

footer_label = ctk.CTkLabel(
    footer_frame, text="Powered by AI | Health Monitoring System", text_color="gray"
)
footer_label.pack(pady=10)

root.mainloop()
