import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load datasets
train = pd.read_csv("train - Walmart Sales Forecast.csv")
features = pd.read_csv("features - Walmart Sales Forecast.csv")
stores = pd.read_csv("stores - Walmart Sales Forecast.csv")

# Convert 'Date' to datetime
train['Date'] = pd.to_datetime(train['Date'])
features['Date'] = pd.to_datetime(features['Date'])

# Merge train with features
df = pd.merge(train, features, how='left', on=['Store', 'Date', 'IsHoliday'])

# Merge with stores metadata
df = pd.merge(df, stores, how='left', on='Store')

# Fill missing MarkDowns with 0
markdown_cols = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
df[markdown_cols] = df[markdown_cols].fillna(0)

# Feature engineering
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Week'] = df['Date'].dt.isocalendar().week

# Convert IsHoliday to int
df['IsHoliday'] = df['IsHoliday'].astype(int)

# Encode 'Type'
le = LabelEncoder()
df['Type'] = le.fit_transform(df['Type'])

# Drop date and define features/target
X = df.drop(['Weekly_Sales', 'Date'], axis=1)
y = df['Weekly_Sales']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
model = RandomForestRegressor(n_estimators=10, random_state=42, n_jobs=-1)

model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"✅ Model Evaluation:")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.4f}")



import matplotlib.pyplot as plt
import seaborn as sns

feature_importance = model.feature_importances_
feature_names = X.columns

# Plot top features
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title("Top Feature Importances")
plt.show()

import joblib
joblib.dump(model, "walmart_sales_model.pkl")
print("✅ Model saved successfully.")

