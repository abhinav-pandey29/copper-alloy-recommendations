# Import libraries
from warnings import filterwarnings
import itertools
import pickle
import os

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

filterwarnings('ignore')
pd.options.display.max_columns = 999

ROOTDIR = os.path.abspath(os.curdir)


def load_data_for_prop(prop):

    prop_paths = {
        'tensile_strength': ROOTDIR + "/data/Tensile Strength Data.csv",
        'thermal_conductivity': ROOTDIR + "/data/Thermal Conductivity Data.csv",
    }

    data = pd.read_csv(prop_paths[prop])
    return data


class ThermalConductivityPredictor:
    def __init__(self, model_path: str, scaler_path: str):
        # Load the scaler and model
        self.scaler = joblib.load(scaler_path)
        self.model = joblib.load(model_path)

    def preprocess(self, data: pd.DataFrame) -> np.ndarray:
        # Preprocess the new data using the loaded scaler
        data_scaled = self.scaler.transform(data)
        return data_scaled

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        # Preprocess the data
        data_scaled = self.preprocess(data)
        # Make predictions
        predictions = self.model.predict(data_scaled)
        return predictions

def load_pretrained_model(prop):
    model_info = {
        'tensile_strength': {'fpath': ROOTDIR + '/models/rf_tensile_strength.pickle',
                             'accuracy': 0.924536},
        "thermal_conductivity": {
            "fpath": ROOTDIR + "/trained_models/gbr_model_thermal_conductivity.pkl",
            "accuracy": 0.723626,
        },
    }

    model_filepath = model_info[prop]["fpath"]
    model_accuracy = model_info[prop]["accuracy"]

    if prop == "tensile_strength":
        with open(model_filepath, "rb") as f:
            print(model_filepath)
            model = pickle.load(f)
    elif prop == "thermal_conductivity":
        model = ThermalConductivityPredictor(
            model_path=model_filepath,
            scaler_path=ROOTDIR + "/trained_models/scaler_thermal_conductivity.pkl",
        )
        model_accuracy = model_info[prop]["accuracy"]

    return model, model_accuracy


def generate_synthetic_data(target_property, target_value, size_limit=30000):
    dataset = load_data_for_prop(target_property)

    # Select the top 10 closest matching alloys
    sorted_alloys = dataset.sort_values(
        by=target_property,
        key=lambda x: abs(target_value - x),
        ascending=True,
    )
    similar_alloys = sorted_alloys.head(10).drop(target_property, axis=1)

    # Generate unique element percentages and all possible combinations,
    # enforcing size limit to prevent combinatorial explosion
    unique_element_percentages = similar_alloys.apply(np.unique)
    all_combinations = itertools.product(*unique_element_percentages.values)
    subset_combinations = list(itertools.islice(all_combinations, size_limit))

    synthetic_alloys = pd.DataFrame(subset_combinations, columns=similar_alloys.columns)

    # Normalize the element percentages to ensure each row sums to 100%
    numeric_columns = synthetic_alloys.select_dtypes(include=[np.number]).columns
    numeric_columns_sum = synthetic_alloys[numeric_columns].sum(axis=1)
    synthetic_alloys[numeric_columns] = (
        synthetic_alloys[numeric_columns].div(numeric_columns_sum, axis=0).multiply(100)
    )

    return synthetic_alloys


def calculate_confidence(predictions, desired_value, model_accuracy):
    percentage_error = abs(predictions - desired_value) / desired_value * 100
    confidence = (100 - percentage_error) * model_accuracy
    return confidence


def run_inverse_model(prop, value, n):
    data = load_data_for_prop(prop)
    exact_matches = data[data[prop] == value].drop(prop, axis=1).drop_duplicates()
    exact_matches.insert(0, "Confidence %", 100)

    model, accuracy = load_pretrained_model(prop)
    synthetic_data = generate_synthetic_data(prop, value)
    predictions = model.predict(synthetic_data)

    synthetic_data["Confidence %"] = calculate_confidence(predictions, value, accuracy)

    alloy_compositions = (
        pd.concat([exact_matches, synthetic_data])
        .drop_duplicates()
        .sort_values(by="Confidence %", ascending=False)
        .reset_index(drop=True)
        .head(n)
    )
    return alloy_compositions
