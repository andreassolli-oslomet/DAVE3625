# Ruter-trip

# DAVE3625 Assignment 2: Machine Learning
## Case 2: Predict passenger data for Ruter
Predict passenger data for Ruter. Use the same data set given to you in assignment 1. I want you to make a prediction algorithm which predicts the number of passengers on a specific date for a specific bus (pick any one). Input should be date and output will be number of passengers. You should also show the prediction percentage score. 

## Findings and Conclusion:
### Why we chose regression: 
We decided upon using regression for our algorithm as it is better suited to calculating basic numbers based on features and data that is entered into the algorithm, such as the passenger count, dates and the bus route. Together these form a relationship which can be used to generate a predicted value.

### Findings
The type of regression model we chose was Poisson, reason being that it is well suited for count data such as passenger count. Poisson also uses a minimum value of 0, which in this case is good as the passenger count on a bus cannot realistically be below 0. Through training our machine learning model we also found that including other features into it such as hours and more accurate times caused the output to be more realistic based on the dataset. With that said we found that our MSE score was quite high, usually around 20-30 even though the predicted passengers seemed realistic. The model´s output could be unrealistic based upon this error margin. There are many factors to why the MSE is so high: Model choice(Could be a better model out there), features(Perhaps need more/better quality features) or quality of data, just to mention a few factors. 

## Group Members: 
* Andreas Sandvik Solli (s364747)
* Christoffer Christensen Naug (s364714)
* Eirik Jørgensen (s358857)
