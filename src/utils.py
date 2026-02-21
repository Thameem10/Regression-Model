import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

def load_data(filename):
    BASE_DIR = Path(__file__).resolve().parent.parent
    data_path = BASE_DIR / "data" / "processed" / filename
    df = pd.read_csv(data_path)
    return df


def split_data(df, target_column, test_size=0.2):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=42,
    )

    return X_train, X_test, y_train, y_test