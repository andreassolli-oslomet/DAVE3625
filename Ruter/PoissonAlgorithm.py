import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

# Load and preprocess the dataset
ruter_data = pd.read_csv('ruter-data.csv', delimiter=';')
ruter_data['Dato'] = pd.to_datetime(ruter_data['Dato'], format='%d/%m/%Y')
ruter_data['DayOfWeek'] = ruter_data['Dato'].dt.dayofweek
ruter_data['Month'] = ruter_data['Dato'].dt.month
ruter_data['Year'] = ruter_data['Dato'].dt.year
ruter_data['Day'] = ruter_data['Dato'].dt.day
ruter_data['Avgang_Hour'] = pd.to_datetime(ruter_data['Tidspunkt_Faktisk_Avgang_Holdeplass_Fra'], format='%H:%M:%S', errors='coerce').dt.hour
ruter_data.dropna(subset=['Avgang_Hour'], inplace=True)

# Select a specific bus line
specific_bus = '380'  # Replace with your chosen bus line
ruter_data_specific = ruter_data[ruter_data['Linjenavn'] == specific_bus]

# Selecting features and target
X = ruter_data_specific[['DayOfWeek', 'Month', 'Year', 'Day', 'Avgang_Hour']]
y = ruter_data_specific['Passasjerer_Ombord']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y = y.clip(lower=0)


# Train the model on the entire dataset
model_poisson = PoissonRegressor()
model_poisson.fit(X, y)

y_pred = model_poisson.predict(X)

# Calculate R² and MSE
r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)


# Plotting actual vs. predicted passenger counts
plt.figure(figsize=(10, 6))
plt.scatter(y, y_pred, alpha=0.5)
plt.xlabel('Actual Passenger Count')
plt.ylabel('Predicted Passenger Count')
plt.title(f'Actual vs. Predicted Passenger Count for Bus Line 390\nR² Score: {r2:.2f}, MSE: {mse:.2f}')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')  # Diagonal line
plt.show()


# Prediction function
def predict_passengers(date_str, time_hour, model):
    date = pd.to_datetime(date_str, format='%Y-%m-%d')
    features = pd.DataFrame({
        'DayOfWeek': [date.dayofweek],
        'Month': [date.month],
        'Year': [date.year],
        'Day': [date.day],
        'Avgang_Hour': [time_hour]
    })
    prediction = model.predict(features)
    return prediction[0]

def predict_passengers_with_confidence(date_str, time_hour, model, std_residuals):
    prediction = predict_passengers(date_str, time_hour, model)
    lower_bound = max(prediction - std_residuals, 0)  # Ensure non-negative
    upper_bound = prediction + std_residuals
    return prediction, lower_bound, upper_bound

y_train_pred = model_poisson.predict(X)
residuals = y - y_train_pred


std_residuals = np.std(residuals)

# Example: Predict passengers for a specific date and time
predicted_passengers, lower_bound, upper_bound = predict_passengers_with_confidence('2023-11-01', 7, model_poisson, std_residuals)  # 14 represents 2 PM
print(f"Predicted number of passengers: {predicted_passengers}")
print(f"Confidence Interval: {lower_bound} to {upper_bound}")
print(f"R-squared (R²) score: {r2}")
print(f"Mean Squared Error (MSE): {mse}")
