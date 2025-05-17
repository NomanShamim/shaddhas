import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import median_absolute_error
from xgboost import XGBRegressor

# Load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')  # assuming test.csv is in same folder

# Drop 'id' column
test_ids = test['id']
train = train.drop(columns=['id'])
test = test.drop(columns=['id'])

# Split features and target
X = train.drop(columns=['Hardness'])
y = train['Hardness']

# Split for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Customized XGBoost model
model = XGBRegressor(
    objective='reg:absoluteerror',   # aligns with MedAE
    tree_method='hist',              # fast + handles skewed features well
    enable_categorical=False,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Evaluate using MedAE
val_preds = model.predict(X_val)
medae = median_absolute_error(y_val, val_preds)
print(f"📊 Median Absolute Error (Validation): {medae:.4f}")

# Retrain on full training data
model.fit(X, y)

# Predict on test set
test_preds = model.predict(test)

# Create submission
submission = pd.DataFrame({
    'id': test_ids,
    'Hardness': test_preds.round(3)
})
submission.to_csv('submission.csv', index=False)

print("✅ submission.csv has been created using optimized XGBoost.")
