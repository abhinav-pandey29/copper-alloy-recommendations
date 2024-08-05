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


def calculate_prediction_confidence(unlabelled_data, desired_value, prop):

    model, model_accuracy = load_pretrained_model(prop)
    predictions = model.predict(unlabelled_data)

    percentage_error = abs(predictions - desired_value) / desired_value * 100
    percentage_confidence = 100 - percentage_error

    prediction_confidence = percentage_confidence * model_accuracy

    return prediction_confidence


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
    unique_element_percentages = df.apply(np.unique)
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


def fetch_exact_matches(prop, value):
    data = load_data_for_prop(prop)

    exact_matches = data[data[prop] == value]
    num_exact_matches = len(exact_matches)

    if num_exact_matches:

        exact_matches.drop(prop, axis=1, inplace=True)
        exact_matches.drop_duplicates(inplace=True)
        exact_matches.insert(
            loc=0,
            column="Confidence %",
            value=[100]*len(exact_matches)
        )

        return exact_matches


def load_data_encoders(prop):

    if prop == 'tensile_strength':
        data = load_data_for_prop(prop)
        form_enc = LabelEncoder().fit(data['Form'])
        temper_enc = LabelEncoder().fit(data['Temper'])

        encoders = {
            'tensile_strength': {
                'Form': form_enc,
                'Temper': temper_enc
            },
            'thermal_conductivity': {}
        }

        return encoders


def decode_temper_form(data, prop):

    encoders = load_data_encoders(prop)
    if encoders:
        if prop == 'tensile_strength':
            data.loc[:, 'Form'] = encoders[prop]['Form'].inverse_transform(
                data['Form']
            )
            data.loc[:, 'Temper'] = encoders[prop]['Temper'].inverse_transform(
                data['Temper']
            )
    return data


def prop_mapper(property_name):

    prop_mapping = {'Tensile Strength': 'tensile_strength',
                    'Thermal Conductivity': 'thermal_conductivity'}
    if property_name in prop_mapping.keys():
        prop = prop_mapping[property_name]

    return prop


def run_inverse_model(prop, value, n):

    prop = prop_mapper(prop)

    exact_matches = fetch_exact_matches(prop, value)

    model, model_accuracy = load_pretrained_model(prop)
    encoders = load_data_encoders(prop)

    synthetic_data = generate_synthetic_data(value, prop)

    if encoders:
        synthetic_data['Form'] = encoders[prop]['Form'].transform(
            synthetic_data['Form']
        )
        synthetic_data['Temper'] = encoders[prop]['Temper'].transform(
            synthetic_data['Temper']
        )

    predictions = model.predict(synthetic_data)

    promising_results = synthetic_data.iloc[np.argsort(
        abs(predictions - value)), :]
    percentage_confidence = calculate_prediction_confidence(
        promising_results, value, prop)

    promising_results.insert(loc=0, column="Confidence %",
                             value=percentage_confidence.round(1), allow_duplicates=False)
    promising_results = decode_temper_form(promising_results, prop)

    results = pd.concat((exact_matches, promising_results), axis=0)
    results = results.loc[:, (results != 0).any(axis=0)]
    results.drop_duplicates(inplace=True)
    results = results.sort_values(by='Confidence %', ascending=False)
    results.reset_index(inplace=True, drop=True)

    return results.round(3).head(n)
