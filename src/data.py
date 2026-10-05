from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "creditcard.csv"
RANDOM_STATE = 42


def load_split():
    """Return a 60/20/20 stratified train/validation/test split."""
    df = pd.read_csv(DATA_PATH)
    df = df.drop_duplicates()   # identical rows could land in both train and test
    df = df.drop(columns=["Time"])  # seconds since the first transaction; not meaningful outside this 2-day file
    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=RANDOM_STATE)
    return X_train, X_val, X_test, y_train, y_val, y_test