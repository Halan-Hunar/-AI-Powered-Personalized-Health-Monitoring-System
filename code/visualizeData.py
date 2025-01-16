import matplotlib.pyplot as plt
from code.handlingDatasets import load_and_preprocess_data
import pandas as pd

# Load datasets
heart_rate_data = load_and_preprocess_data()
bmi_data = pd.read_csv('bmi_data_200.csv')
blood_sugar_data = pd.read_csv('blood_sugar_data.csv')

# Visualize heart rate trends
def visualize_heart_rate():
    if heart_rate_data.empty:
        print("Error: No heart rate data available to visualize.")
        return
    plt.plot(heart_rate_data['heart_rate'], label='Heart Rate')
    plt.title('Heart Rate Trends')
    plt.xlabel('Data Points')
    plt.ylabel('Heart Rate (Scaled)')
    plt.legend()
    plt.show()

# Visualize BMI distribution
def visualize_bmi():
    if bmi_data.empty:
        print("Error: No BMI data available to visualize.")
        return
    bmi_data['BMI'].hist(bins=20, edgecolor='black')
    plt.title('BMI Distribution')
    plt.xlabel('BMI')
    plt.ylabel('Frequency')
    plt.show()

# Visualize blood sugar distribution
def visualize_blood_sugar():
    if blood_sugar_data.empty:
        print("Error: No blood sugar data available to visualize.")
        return
    blood_sugar_data['Fasting Blood Sugar (mg/dL)'].hist(bins=20, edgecolor='black')
    plt.title('Fasting Blood Sugar Distribution')
    plt.xlabel('Fasting Blood Sugar (mg/dL)')
    plt.ylabel('Frequency')
    plt.show()
