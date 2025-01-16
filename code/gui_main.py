import customtkinter as ctk
import joblib
from code.model import train_and_save_model
from code.visualizeData import visualize_data
from code.handlingDatasets import load_and_preprocess_data

# Function to load the trained model
def load_model():
    try:
        model = joblib.load('health_model.pkl')  # Load the trained model
        return model
    except FileNotFoundError:
        ctk.CTkMessagebox.show_error("Model Error", "Trained model not found! Please train the model first.")
        return None

# Function to normalize user-input heart rate
def normalize_input_heart_rate(value):
    min_heart_rate = 60  # Minimum heart rate from dataset
    max_heart_rate = 160  # Maximum heart rate from dataset
    return (value - min_heart_rate) / (max_heart_rate - min_heart_rate)

# Function to predict risk level
def predict_risk_level_direct(heart_rate, activity_level):
    model = load_model()
    if model:
        try:
            # Create input for prediction
            input_data = [[heart_rate, activity_level]]
            prediction = model.predict(input_data)
            return prediction[0]
        except Exception as e:
            ctk.CTkMessagebox.show_error("Prediction Error", f"Error during prediction: {str(e)}")
            return None
    else:
        return None

# Function to train the model
def train_model():
    train_and_save_model()
    ctk.CTkMessagebox.show_info("Training Complete", "Model training completed successfully!")

# Function to visualize data
def visualize():
    visualize_data()
    ctk.CTkMessagebox.show_info("Visualization Complete", "Health data visualization completed!")

# Function to predict risk level via GUI
def predict():
    def get_prediction():
        try:
            heart_rate = float(heart_rate_entry.get())  # Get heart rate input
            activity_level = float(activity_level_entry.get())  # Get activity level input

            # Normalize heart rate
            heart_rate_scaled = normalize_input_heart_rate(heart_rate)

            # Validate inputs
            if not (0 <= heart_rate_scaled <= 1) or not (0 <= activity_level <= 1):
                ctk.CTkMessagebox.show_error("Input Error", "Heart rate must be between 60-160 and activity level between 0-1.")
                return

            # Get prediction
            prediction = predict_risk_level_direct(heart_rate_scaled, activity_level)
            if prediction:
                # Display prediction
                if prediction.upper() == "HIGH":
                    result_label.configure(text="🔴 HIGH RISK", text_color="red", font=("Helvetica", 18, "bold"))
                elif prediction.upper() == "MEDIUM":
                    result_label.configure(text="🟠 NORMAL", text_color="lighblue", font=("Helvetica", 18, "bold"))
                else:
                    result_label.configure(text="🟢 LOW RISK", text_color="green", font=("Helvetica", 18, "bold"))

                details_label.configure(
                    text=f"Heart Rate (scaled): {heart_rate_scaled:.2f}\nActivity Level: {activity_level}",
                    font=("Helvetica", 14),
                    justify="left"
                )
            else:
                result_label.configure(text="Error: Could not make prediction.", text_color="red")
                details_label.configure(text="")
        except ValueError:
            ctk.CTkMessagebox.show_error("Input Error", "Please enter valid numeric values.")

    # New window for prediction inputs
    prediction_window = ctk.CTkToplevel(root)
    prediction_window.title("Predict Risk Level")
    prediction_window.geometry("400x500")

    # Input fields
    ctk.CTkLabel(prediction_window, text="Enter Heart Rate (60-160):", font=("Helvetica", 14)).pack(pady=10)
    heart_rate_entry = ctk.CTkEntry(prediction_window, width=300)
    heart_rate_entry.pack(pady=5)

    ctk.CTkLabel(prediction_window, text="Enter Activity Level (0-1):", font=("Helvetica", 14)).pack(pady=10)
    activity_level_entry = ctk.CTkEntry(prediction_window, width=300)
    activity_level_entry.pack(pady=5)

    # Predict button
    ctk.CTkButton(prediction_window, text="Predict", command=get_prediction, width=150).pack(pady=15)

    # Result display
    result_label = ctk.CTkLabel(prediction_window, text="", font=("Helvetica", 18))
    result_label.pack(pady=10)

    details_label = ctk.CTkLabel(prediction_window, text="", font=("Helvetica", 14), justify="left")
    details_label.pack(pady=5)

# GUI setup
ctk.set_appearance_mode("System")  # Modes: "System" (default), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue", "green", "dark-blue"

root = ctk.CTk()
root.title("AI-Powered Health Monitoring System")
root.geometry("600x500")
root.resizable(True, True)

# Main title
ctk.CTkLabel(root, text="Welcome to the Health Monitoring System", font=("Helvetica", 20)).pack(pady=20)

# Buttons
train_button = ctk.CTkButton(root, text="Train Model", command=train_model, width=200, height=40, font=("Helvetica", 14))
train_button.pack(pady=15)

visualize_button = ctk.CTkButton(root, text="Visualize Data", command=visualize, width=200, height=40, font=("Helvetica", 14))
visualize_button.pack(pady=15)

predict_button = ctk.CTkButton(root, text="Predict Risk Level", command=predict, width=200, height=40, font=("Helvetica", 14))
predict_button.pack(pady=15)

exit_button = ctk.CTkButton(root, text="Exit", command=root.quit, width=200, height=40, font=("Helvetica", 14), fg_color="red")
exit_button.pack(pady=15)

root.mainloop()
