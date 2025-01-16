from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import pandas as pd

# Load BMI dataset
bmi_data = pd.read_csv('bmi_data_200.csv')

# Features and target
X = bmi_data[['Height (m)', 'Weight (kg)']]
y = bmi_data['Category']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
bmi_model = RandomForestClassifier(n_estimators=50, random_state=42)
bmi_model.fit(X_train, y_train)

# Evaluate the model
train_accuracy = bmi_model.score(X_train, y_train)
test_accuracy = bmi_model.score(X_test, y_test)
print(f"BMI Model - Training Accuracy: {train_accuracy:.2f}, Test Accuracy: {test_accuracy:.2f}")

# Save the model
joblib.dump(bmi_model, 'bmi_model.pkl')
print("BMI model saved as 'bmi_model.pkl'.")
