# Import libraries
from warnings import filterwarnings
import itertools
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

filterwarnings('ignore')
pd.options.display.max_columns = 999


def extract_unique_values(data):
    return [np.unique(data[col]) for col in data]


def load_data_for_prop(prop):

    prop_paths = {
        'tensile_strength': "data/Tensile Strength Data.csv",
        'thermal_conductivity': "data/Thermal Conductivity Data.csv",
    }

    data = pd.read_csv(prop_paths[prop])
    return data


def extract_closest_matches(prop, value, n):
    '''
    Funcion to get n alloys with the closest values of the
    desired property (thermal conductivity or tensile strength)
    '''

    data = load_data_for_prop(prop)

    abs_diff = abs(data[prop] - value)
    top_matches_idx = np.argsort(abs_diff)[:n]

    top_matches = data.iloc[top_matches_idx]
    top_matches.drop(prop, axis=1, inplace=True)

    return top_matches


def limit_synthetic_data_size(data, prop, value, size_limit=300000):

    for n in range(len(data), 1, -1):
        synthetic_data_content = extract_unique_values(
            extract_closest_matches(prop, value, n)
        )

        estimated_size_of_dataset = np.prod(
            [len(ele) for ele in synthetic_data_content])

        if estimated_size_of_dataset <= size_limit:
            size_reduced_matches = extract_closest_matches(prop, value, n)
            break

    return size_reduced_matches


def extract_permutations(list_of_lists):
    return list(itertools.product(*list_of_lists))


def rectify_composition(data):
    return data.apply(lambda x: 100 * x / data.sum(axis=1))


def load_pretrained_model(prop):
    model_info = {
        'tensile_strength': {'fpath': 'models/rf_tensile_strength.pickle',
                             'accuracy': 0.924536},
        'thermal_conductivity': {'fpath': 'models/rf_thermal_conductivity.pickle',
                                 'accuracy': 0.828387},
    }

    model_filepath = model_info[prop]['fpath']
    model_accuracy = model_info[prop]['accuracy']

    with open(model_filepath, 'rb') as f:
        model = pickle.load(f)

    return model, model_accuracy


def calculate_prediction_confidence(unlabelled_data, desired_value, prop):

    model, model_accuracy = load_pretrained_model(prop)
    predictions = model.predict(unlabelled_data)

    percentage_error = abs(predictions - desired_value) / desired_value * 100
    percentage_confidence = 100 - percentage_error

    prediction_confidence = percentage_confidence * model_accuracy

    return prediction_confidence


def generate_synthetic_data(desired_value, prop):

    top_matches = extract_closest_matches(prop, desired_value, n=10)
    top_matches = limit_synthetic_data_size(
        top_matches, prop, desired_value, size_limit=300000)
    columns = top_matches.columns

    components_of_top_matches = extract_unique_values(top_matches)
    component_permutations = extract_permutations(components_of_top_matches)

    generated_dataset = pd.DataFrame(component_permutations, columns=columns)
    generated_dataset.loc[:, 'Cu':] = rectify_composition(
        generated_dataset.loc[:, 'Cu':])

    return generated_dataset


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
    results = results.sort_values(by='Confidence %', ascending=False)
    results.reset_index(inplace=True, drop=True)

    return results.round(3).head(n)
