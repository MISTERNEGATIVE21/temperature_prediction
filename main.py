import numpy as np
from sklearn.linear_model import LinearRegression

# Collect weather data for the previous day and time
temperatures = [293.15, 283.15, 272.15, 266.15, 274.15]
humidity = [0.8, 0.7, 0.65, 0.6, 0.63]
times = [12, 15, 18, 21, 0]

# Convert data to numpy arrays for use with scikit-learn
X = np.array([[temp, time] for temp, time in zip(temperatures, times)])
y = np.array(humidity)

# Train a linear regression model
model = LinearRegression()
model.fit(X, y)

# Use the trained model to make a prediction for the current day and time
prediction = model.predict([[277.15, 6]])[0]
print("Predicted humidity for today at 6am:", prediction)
