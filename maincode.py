import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv('HousingDataset.csv')
# Assuming 'data' contains the dataset

# Separate features and target variable
X = data.drop('price', axis=1)
y = data['price']

# Encode categorical variables
encoder = LabelEncoder()
X_categorical = X.select_dtypes(include=['object'])
X_categorical = X_categorical.apply(encoder.fit_transform)

# Combine categorical and numerical features
X.drop(X_categorical.columns, axis=1, inplace=True)
X = pd.concat([X, X_categorical], axis=1)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)