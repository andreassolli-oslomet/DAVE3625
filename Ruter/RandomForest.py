
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load and preprocess the dataset
ruter_data = pd.read_csv("https://raw.githubusercontent.com/atikagondal/Assignment-2-dave3625-202323/main/Ruter-data.csv", delimiter=';')
ruter_data['Dato'] = pd.to_datetime(ruter_data['Dato'], format='%d/%m/%Y')
ruter_data['DayOfWeek'] = ruter_data['Dato'].dt.dayofweek
ruter_data['Month'] = ruter_data['Dato'].dt.month
ruter_data['TimeOfDay'] = pd.to_datetime(ruter_data['Tidspunkt_Faktisk_Avgang_Holdeplass_Fra'], format='%H:%M:%S', errors='coerce').dt.hour
ruter_data['TimeOfDay'] = ruter_data['TimeOfDay'].fillna(ruter_data['TimeOfDay'].mean())  # Replace NaN with mean

# Select a specific bus line
specific_bus = '390'  # Focusing on bus line 21 for this example
ruter_data_specific = ruter_data[ruter_data['Linjenavn'] == specific_bus]

# Selecting features and target
X = ruter_data_specific[['DayOfWeek', 'Month', 'Kjøretøy_Kapasitet', 'TimeOfDay']]
y = ruter_data_specific['Passasjerer_Ombord']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model on the dataset
model = RandomForestRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


# Prediction function with post-processing to handle negative predictions
def predict_passengers(date_str, time_str, model):
    date = pd.to_datetime(date_str, format='%Y-%m-%d')
    time = pd.to_datetime(time_str, format='%H:%M')
    day_of_week = date.dayofweek
    month = date.month
    hour = time.hour
    avg_bus_capacity = X['Kjøretøy_Kapasitet'].mean()  # Using the average bus capacity
    features = pd.DataFrame({
        'DayOfWeek': [day_of_week],
        'Month': [month],
        'Kjøretøy_Kapasitet': [avg_bus_capacity],
        'TimeOfDay': [hour]
    })
    prediction = model.predict(features)
    return max(0, prediction[0])  # Ensuring the prediction is not negative

# Example usage
predicted_passengers = predict_passengers("2023-11-01", "07:00", model)
print(f"Predicted number of passengers: {predicted_passengers}")


plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Actual Passenger Count')
plt.ylabel('Predicted Passenger Count')
plt.title('Actual vs. Predicted Passenger Count')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')  # Diagonal line
plt.show()
