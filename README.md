# Ruter-trip

We chose the task about predicting the amount of passangers on a spesific bus. The data was gathered from a dataset provided by ruter. 
Choosing Poisson Regressor for Ruter Data Analysis

1. Poisson Regressor is well-suited for modeling count data like passenger numbers, as it directly models a count output based on input features like date and bus line, assuming the mean and variance of the distribution are equal.
2. This model should be effective for this dataset because passenger counts are typically non-negative and discrete, matching the Poisson distribution's characteristics.
3. The inclusion of time-related features (like DayOfWeek, Month, and Hour) in the model allows us to capture time-based patterns in passenger counts, critical for accurate predictions in the public transport picture.
4. We understand that the MSE is high and for this reason the model has a lot of room for improvement. The model´s output could be unrealistic based upon this error margin. 
