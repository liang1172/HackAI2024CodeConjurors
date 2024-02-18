import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt

# Load the dataset
data = pd.read_csv('HousingDataset.csv')
# Assuming 'data' contains the dataset

# Separate features and target variable
X = data.drop("price", axis=1)
y = data["price"]

# Encode categorical variables
encoder = LabelEncoder()
X_categorical = X.select_dtypes(include=['object'])
X_categorical = X_categorical.apply(encoder.fit_transform)

# Combine categorical and numerical features
X.drop(X_categorical.columns, axis=1, inplace=True)
X = pd.concat([X, X_categorical], axis=1)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Linear Regression model
model = LinearRegression()
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(max_depth=2, random_state=42)
rf.fit(X_train_scaled, y_train)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)
rfy_pred = rf.predict(X_test_scaled)

# Measure the performance of the model
score = model.score(X_test_scaled, y_test).round(3)
rfscore = rf.score(X_test_scaled,y_test).round(3)
print("LRTest Score:", score)
print("RFTest Score:", rfscore)
score1 = model.score(X_train_scaled, y_train).round(3)
rfscore1 = rf.score(X_train_scaled, y_train).round(3)
print("LRTraining Score:", score1)
print("RFTrainging Score:", rfscore1)

# Evaluate the model
mse = math.sqrt(mean_squared_error(y_test, y_pred))
rfmse = math.sqrt(mean_squared_error(y_test, rfy_pred))
print("LRMean Squared Error:", mse)
print("RFMean Squared Error:", rfmse)
# box and whisker plots
data.plot(kind='box', subplots=True, layout=(13,13), sharex=False, sharey=False)
plt.show()

data_new = X_test_scaled[0:5]

pred = model.predict(data_new)
rfpred = rf.predict(data_new)
print("LRPredict:", pred)
print("RFPredict:", rfpred)
print("True output:", y_test[0:5])
