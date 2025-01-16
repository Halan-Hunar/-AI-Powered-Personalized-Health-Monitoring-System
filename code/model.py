from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from code.handlingDatasets import load_and_preprocess_data

# Load and preprocess the data
data = load_and_preprocess_data()

def train_and_save_model():
    # Split data into features and target
    X = data[['heart_rate', 'activity_level']]  # Scaled features
    y = data['risk_level']  # Target

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    print(f"Training Accuracy: {train_accuracy:.2f}")
    print(f"Test Accuracy: {test_accuracy:.2f}")

    # Test predictions for debugging
    sample_inputs = [[0.2, 0.3], [0.6, 0.5], [0.9, 0.8]]
    print("Sample Predictions:", model.predict(sample_inputs))

    # Save model
    joblib.dump(model, 'health_model.pkl')
    print("Model saved as 'health_model.pkl'.")
