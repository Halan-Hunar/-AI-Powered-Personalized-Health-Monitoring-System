import matplotlib.pyplot as plt
from code.handlingDatasets import load_and_preprocess_data

# Load and preprocess data
data = load_and_preprocess_data()

# Visualize heart rate trends
def visualize_data():
    if data.empty:
        print("Error: No data available to visualize.")
    else:
        plt.plot(data['heart_rate'], label='Heart Rate')
        plt.title('Heart Rate Trends')
        plt.xlabel('Data Points')
        plt.ylabel('Heart Rate (Scaled)')
        plt.legend()
        plt.show()
