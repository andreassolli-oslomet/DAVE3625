# Ruter-trip

Choosing Poisson Regressor for Ruter Data Analysis

1. Poisson Regressor is well-suited for modeling count data like passenger numbers, as it directly models a count outcome based on input features like date and bus line, assuming the mean and variance of the distribution are equal.
2. This model is effective for this dataset because passenger counts are typically non-negative and discrete, matching the Poisson distribution's characteristics.
3. The inclusion of time-related features (like DayOfWeek, Month, and Hour) in the model allows us to capture time-based patterns in passenger counts, critical for accurate predictions in the public transport picture.

4. We also provided another file, Random Forest regressor. We added this because we were not shure which of them gave the best result. There are a few descrepencies between the two in the plot chart. 
