import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Sample dataset: Height (in feet) vs Weight (in kg)
data = {
    'Height': [5.0, 5.2, 5.4, 5.6, 5.8, 6.0, 6.2],
    'Weight': [50, 54, 58, 63, 67, 72, 77]
}

df = pd.DataFrame(data)

# Prepare features and target
X = df[['Height']]
y = df['Weight']

# Train the model
model = LinearRegression()
model.fit(X, y)

# Save the model to a file
joblib.dump(model, 'weight_predictor.pkl')

print("Model trained and saved as weight_predictor.pkl")
