from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import pandas as pd

# Load blood sugar dataset
blood_sugar_data = pd.read_csv('blood_sugar_data.csv')

# Features and target
X = blood_sugar_data[['Fasting Blood Sugar (mg/dL)']]
y = blood_sugar_data['Category']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
blood_sugar_model = RandomForestClassifier(n_estimators=50, random_state=42)
blood_sugar_model.fit(X_train, y_train)

# Evaluate the model
train_accuracy = blood_sugar_model.score(X_train, y_train)
test_accuracy = blood_sugar_model.score(X_test, y_test)
print(f"Blood Sugar Model - Training Accuracy: {train_accuracy:.2f}, Test Accuracy: {test_accuracy:.2f}")

# Save the model
joblib.dump(blood_sugar_model, 'blood_sugar_model.pkl')
print("Blood sugar model saved as 'blood_sugar_model.pkl'.")
