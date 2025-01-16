import joblib

def predict_risk_level():
    try:
        # Load the trained model
        model = joblib.load('health_model.pkl')
        print("Model loaded successfully!")
    except FileNotFoundError:
        print("Error: 'health_model.pkl' not found. Please train the model first.")
        return

    # Take user input
    try:
        heart_rate = float(input("Enter heart rate (scaled between 0 and 1): "))
        activity_level = float(input("Enter activity level (scaled between 0 and 1): "))

        # Predict the risk level
        prediction = model.predict([[heart_rate, activity_level]])
        print(f"Predicted risk level: {prediction[0]}")
    except Exception as e:
        print(f"Error: {e}")
