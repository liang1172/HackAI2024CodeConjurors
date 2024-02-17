import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Load the dataset
file_path = './HousingDataset.csv'
data = pd.read_csv(file_path)

# Define categorical features
categorical_features = ['guestroom', 'mainroad', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']

# Initialize ColumnTransformer
column_transformer = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), categorical_features)], remainder='passthrough')

# Fit and transform the data
X = data.drop('price', axis=1)  # Assuming 'price' is the target variable
y = data['price']
X_transformed = column_transformer.fit_transform(X)

# Now that we have fitted the ColumnTransformer, we can get the transformed feature names
feature_names = column_transformer.get_feature_names_out()

# Split the transformed data
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Get the coefficients from the model
coefficients = model.coef_

# Pair feature names with coefficients, sort by absolute value of coefficient
sorted_features_and_coeffs = sorted(zip(feature_names, coefficients), key=lambda x: abs(x[1]), reverse=True)

# Extract the names of the two most predictive features
top_features_indices = [np.where(feature_names == feature)[0][0] for feature, _ in sorted_features_and_coeffs[:2]]
X_top_features = X_train[:, top_features_indices]

# Split the dataset again if necessary or directly use the indices to train a new model
# Note: This step might be redundant if you're focusing on coefficient analysis rather than retraining

# Display the names and coefficients of the two most predictive features
print("Most Predictive Features:")
for feature, coeff in sorted_features_and_coeffs[:2]:
    print(f"{feature}: {coeff}")

# MSE was already calculated for the model trained on the full set of features
# If you wish to train a new model on just the top 2 features, follow a similar procedure as above

# Note: Visualizing specific features' impact on the model would require access to those features' raw values for plotting
