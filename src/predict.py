import joblib
import pandas as pd
from pathlib import Path

# ----------------------------
# Load Model
# ----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
model_path = BASE_DIR / "models" / "gb_regressor_model.pkl"

model = joblib.load(model_path)

print("Model loaded successfully.")

# ----------------------------
# Create New Sample Input
# (Replace values with real data)
# ----------------------------

new_sample = pd.DataFrame(
    [[
        3,      # number of bedrooms
        2,      # number of bathrooms
        1800,   # living area
        5000,   # lot area
        2,      # number of floors
        0,      # waterfront present
        2,      # number of views
        3,      # condition of the house
        7,      # grade of the house
        1500,   # area excluding basement
        300,    # basement area
        2005,   # built year
        0,      # renovation year
        2000,   # living_area_renov
        5200,   # lot_area_renov
        3,      # number of schools nearby
        15      # distance from airport
    ]],
    columns=[
        "number of bedrooms",
        "number of bathrooms",
        "living area",
        "lot area",
        "number of floors",
        "waterfront present",
        "number of views",
        "condition of the house",
        "grade of the house",
        "Area of the house(excluding basement)",
        "Area of the basement",
        "Built Year",
        "Renovation Year",
        "living_area_renov",
        "lot_area_renov",
        "Number of schools nearby",
        "Distance from the airport"
    ]
)

# ----------------------------
# Make Prediction
# ----------------------------

predicted_price = model.predict(new_sample)

print("Predicted House Price:", predicted_price[0])