import joblib
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models"

# 1. Load BOTH the model and the scaler
model = joblib.load(MODEL_PATH / "house_price_model.pkl")
scaler = joblib.load(MODEL_PATH / "house_price_scaler.pkl")
print("Model and Scaler loaded successfully.")

# 2. Create New Sample Input
new_sample = pd.DataFrame(
    [[3, 2, 1800, 2, 7, 1500, 300, 2000, 52.89]],
    columns=[
        "number of bedrooms", "number of bathrooms", "living area", 
        "number of floors", "grade of the house", 
        "Area of the house(excluding basement)", "Area of the basement", 
        "living_area_renov", "Lattitude"
    ]
)

# 3. Fix Feature Order (Just in case)
expected_columns = model.feature_names_in_
new_sample = new_sample[expected_columns]

# 4. SCALE THE NEW SAMPLE!
# Notice we use .transform(), NOT .fit_transform()
# We want to apply the exact same mathematical rules learned during training.
new_sample_scaled = scaler.transform(new_sample)

# 5. Make Prediction & Convert
# Pass the SCALED data to the model
predicted_price_log = model.predict(new_sample_scaled)


print(f"\nPredicted House Price: ${predicted_price_log[0]:,.2f}")