import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import median_absolute_error




# Load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Drop 'id' columns
test_ids = test['id']
train = train.drop(columns=['id'])
test = test.drop(columns=['id'])

# Split features and target
X = train.drop(columns=['Hardness'])
y = train['Hardness']

# Split training data for MedAE validation (optional)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train LightGBM model (no tuning)
model = lgb.LGBMRegressor()
model.fit(X_train, y_train)


# Evaluate on validation set
val_preds = model.predict(X_val)
medae = median_absolute_error(y_val, val_preds)
print(f"📊 Median Absolute Error (Validation): {medae:.4f}")

# Retrain on full training data
model.fit(X, y)

# Predict on test data
test_preds = model.predict(test)

# Save submission file
# Save submission file with rounded predictions
submission = pd.DataFrame({
    'id': test_ids,
    'Hardness': test_preds.round(3)  # Match sample format
})
submission.to_csv('submission.csv', index=False)

print("✅ submission.csv has been created.")
