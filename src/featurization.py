import os
import sys
import yaml
import scipy
import numpy as np
from tqdm import tqdm
import pywt

# calculate entropy for numpy 2d array by first converting it to a pro


def calculate_entropy_2d(array):
    flat_matrix = array.ravel()
    # Get unique values and their frequencies
    _, counts = np.unique(flat_matrix, return_counts=True)
    # Calculate probabilities
    probabilities = counts / flat_matrix.size
    entropy = scipy.stats.entropy(probabilities)
    return entropy


def calculate_statistics(list_values):
    n5 = np.nanpercentile(list_values, 5)
    n25 = np.nanpercentile(list_values, 25)
    n75 = np.nanpercentile(list_values, 75)
    n95 = np.nanpercentile(list_values, 95)
    median = np.nanpercentile(list_values, 50)
    mean = np.nanmean(list_values)
    std = np.nanstd(list_values)
    var = np.nanvar(list_values)
    rms = np.nanmean(np.sqrt(list_values**2))
    return [n5, n25, n75, n95, median, mean, std, var, rms]


def compress(original, level, threshold_percentage, wavelet):
    wavelet_repr = pywt.wavedec2(
        original, wavelet, level=level, mode='periodization')
    c_vals, c_idx = pywt.coeffs_to_array(wavelet_repr)
    absolute_coefficients = np.abs(c_vals).flatten()
    threshold = np.max(absolute_coefficients) * threshold_percentage/100
    c_t = pywt.threshold(c_vals, value=threshold, mode='hard')
    threshold_coefficients = pywt.array_to_coeffs(
        c_t, c_idx, output_format='wavedec2')
    return threshold_coefficients


def extract_coefficients(raw_data, level, threshold, wavelet, num, normalize=False):
    coefficients_arr = []
    for i in range(raw_data.shape[2]):
        subject_data = raw_data[:, :, i]
        transformed = compress(subject_data, level, threshold, wavelet)
        if num == 0:
            transformed_array = transformed[0]
        elif num == -1:
            transformed_array = pywt.coeffs_to_array(
                transformed)[0]  # all coefficients
        else:
            transformed_array = np.vstack(
                (transformed[num][0], transformed[num][1], transformed[num][2]))

        coefficients_arr.append(transformed_array)

    coefficients_arr = np.array(coefficients_arr)
    if normalize:
        coefficients_arr = (coefficients_arr - coefficients_arr.min()) / \
            (coefficients_arr.max() - coefficients_arr.min())
    return coefficients_arr


def extract_features(coefficients):
    all_features = []
    for i in range(coefficients.shape[0]):
        subject_coefficients = coefficients[i, :, :]
        entropy = calculate_entropy_2d(subject_coefficients)
        statistics = calculate_statistics(subject_coefficients)
        features = np.concatenate((statistics, [entropy]))
        all_features.append(features)

    all_features = np.array(all_features)
    return all_features


def convert_labels(labels):
    target = np.zeros(len(labels))
    for i in range(len(labels)):
        if labels[i] <= 0.35:
            target[i] = 0
        elif labels[i] >= 0.7:
            target[i] = 1
        else:
            target[i] = 2
    return target


params = yaml.safe_load(open("params.yaml"))["featurize"]

if len(sys.argv) != 4:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write(
        "\tpython featurization.py transformed-path labels-path output-path\n")
    sys.exit(1)

data_name = sys.argv[1].split("/")[1]
input_folder = os.path.join(sys.argv[1])
labels_folder = os.path.join(sys.argv[2])
output_folder = os.path.join(sys.argv[3])

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

all_labels = []
if data_name == "mi":
    for l in os.listdir(labels_folder):
        y_train = scipy.io.loadmat(os.path.join(labels_folder, l))["y_train"]
        y_test = scipy.io.loadmat(os.path.join(labels_folder, l))["y_test"]
        all_labels = np.concatenate((y_train, y_test), axis=1).reshape(-1)
else:
    for l in os.listdir(labels_folder):
        labels = scipy.io.loadmat(os.path.join(labels_folder, l))["perclos"]
        labels = convert_labels(labels)
        all_labels.append(labels)

model_data = []
if data_name == "mi":
    for f in tqdm(os.listdir(input_folder)):
        data = np.load(os.path.join(input_folder, f))
        coefficients = extract_coefficients(data, params["level"],
                                            params["threshold"], params["wavelet"], params["coefficient_number"])
        features = extract_features(coefficients)
        combined = np.column_stack((features, all_labels))
        model_data.append(combined)
else:
    for i, f in enumerate(tqdm(os.listdir(input_folder))):
        data = np.load(os.path.join(input_folder, f))
        coefficients = extract_coefficients(data, params["level"],
                                            params["threshold"], params["wavelet"], params["coefficient_number"])
        features = extract_features(coefficients)
        combined = np.column_stack((features, all_labels[i]))
        model_data.append(combined)

model_data = np.vstack(model_data)
np.save(os.path.join(output_folder, "wavelet_features"), model_data)
