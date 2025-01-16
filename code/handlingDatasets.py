from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def load_and_preprocess_data(file_path='health_data.csv'):
    # Load dataset
    data = pd.read_csv(file_path)

    # Normalize numeric columns
    scaler = MinMaxScaler()
    data['heart_rate'] = scaler.fit_transform(data[['heart_rate']])  # Scale heart rate
    print("Heart rate scaled between 0 and 1.")

    # Handle missing values only for numeric columns
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())
    print("Missing values handled for numeric columns.")

    print("Data loaded and preprocessed successfully.")
    return data  # Return the processed dataset
