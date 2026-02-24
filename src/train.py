import time
import joblib
import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

from utils import load_data, split_data

# ---------------------------------------
# Load Processed Data using utils
# ---------------------------------------

df = load_data("house_price.csv")

X_train, X_test, y_train, y_test = split_data(df, target_column="Price")

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

# ---------------------------------------
# Model Definition (Random Forest)
# ---------------------------------------

model = RandomForestRegressor(random_state=42)

param_grid = {
    "n_estimators": [100],      
    "max_depth": [6, 8, 10],             # Pushing depth even lower
    "min_samples_split": [20, 30, 50],   # Forcing much larger groups before splitting
    "min_samples_leaf": [10, 15, 20]     # Forcing larger final prediction buckets
}

kfold = KFold(n_splits=3, shuffle=True, random_state=42)  # reduced from 5 to 3 folds

grid = RandomizedSearchCV(                                  # switched from GridSearchCV
    model,
    param_grid,
    n_iter=20,
    cv=kfold,
    scoring="r2",
    n_jobs=-1,
    verbose=1,
    random_state=42
)

# ---------------------------------------
# Train Model (with timing)
# ---------------------------------------

print("\nStarting hyperparameter search...")
start = time.time()

grid.fit(X_train, y_train)

search_time = time.time() - start
print(f"\nHyperparameter Search Time: {search_time:.2f} seconds")

print("Best Parameters:", grid.best_params_)
print("Best CV R2:", grid.best_score_)

# Retrain best model with 300 estimators for better accuracy
print("\nRetraining best model with 300 estimators...")
retrain_start = time.time()

best_params = grid.best_params_.copy()
best_params["n_estimators"] = 300

best_model = RandomForestRegressor(**best_params, random_state=42)
best_model.fit(X_train, y_train)

retrain_time = time.time() - retrain_start
total_time = search_time + retrain_time

print(f"Retraining Time:            {retrain_time:.2f} seconds")
print(f"Total Training Time:        {total_time:.2f} seconds")

# ---------------------------------------
# Evaluate Model
# ---------------------------------------

y_pred_log = best_model.predict(X_test)

y_pred = np.expm1(y_pred_log)
y_test_actual = np.expm1(y_test)

mae = mean_absolute_error(y_test_actual, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
r2 = r2_score(y_test_actual, y_pred)

print("\nFinal Test Performance")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2:   {r2:.4f}")

print("\n---------------------------------------")
print("          MODEL DIAGNOSIS")
print("---------------------------------------")

# 1. Compare Training vs. Testing to check for Overfitting
# We need to predict on the training data to see how well it learned
y_train_pred_log = best_model.predict(X_train)
r2_train = r2_score(y_train, y_train_pred_log)

print(f"Training R2 Score: {r2_train:.4f}")
print(f"Testing R2 Score:  {r2:.4f}")

# Simple diagnosis logic
if (r2_train - r2) > 0.10:
    print("=> Diagnosis: Model might be OVERFITTING. It memorized the training data but struggles on new data.")
elif r2_train < 0.60:
    print("=> Diagnosis: Model might be UNDERFITTING. It's struggling to learn the patterns in the data.")
else:
    print("=> Diagnosis: Model is generalizing well! Training and testing scores are balanced.")


# 2. K-Fold Cross Validation on the final model
print("\nRunning 5-Fold Cross-Validation on Training Data...")
# This trains and tests the model 5 separate times on different slices of data
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)

print(f"CV R2 Scores for each fold: {np.round(cv_scores, 4)}")
print(f"Average CV R2: {cv_scores.mean():.4f}")
print(f"Score Variance (+/-): {cv_scores.std() * 2:.4f}")

if cv_scores.std() > 0.05:
    print("=> Diagnosis: High variance across folds. The model's performance is highly dependent on how the data is split.")
else:
    print("=> Diagnosis: Low variance. The model is stable and reliable!")
    


# ---------------------------------------
# Save Model
# ---------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models"
MODEL_PATH.mkdir(parents=True, exist_ok=True)

joblib.dump(best_model, MODEL_PATH / "house_price_model.pkl")

print("\nModel saved successfully.")
