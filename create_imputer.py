import pathlib
import pickle

import pandas
from sklearn.impute import KNNImputer
from sklearn import model_selection

SALES_PATH = "data/kc_house_data.csv"
HOME_FEATURES = [
    "bedrooms", "bathrooms", "sqft_living", "sqft_lot",
    "floors", "sqft_above", "sqft_basement",
]
OUTPUT_DIR = "model"


def main():
    data = pandas.read_csv(SALES_PATH, usecols=HOME_FEATURES + ["price"])
    x_train, _, _, _ = model_selection.train_test_split(
        data[HOME_FEATURES], data["price"], random_state=42
    )

    imputer = KNNImputer(n_neighbors=5, weights="distance")
    imputer.fit(x_train)

    output_dir = pathlib.Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)
    with open(output_dir / "imputer.pkl", "wb") as f:
        pickle.dump(imputer, f)
    print("Saved imputer.pkl")


if __name__ == "__main__":
    main()
