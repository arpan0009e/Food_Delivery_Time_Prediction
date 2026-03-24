from data_preprocessing import load_and_preprocess
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load data
df = load_and_preprocess('../data/delivery_data.csv')

# Split
X = df.drop('Delivery_Time_min', axis=1)
y = df['Delivery_Time_min']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

# Save column names
import json

with open('../models/columns.json', 'w') as f:
    json.dump(list(X.columns), f)
    
# Save model
with open('../models/model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model trained and saved successfully!")