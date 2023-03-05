import os
import sys
import yaml
import scipy
import numpy as np
from tqdm import tqdm
import pywt


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


def create_features(raw_data, level, threshold, wavelet, num):
    features_arr = []
    for i in range(raw_data.shape[2]):
        subject_data = raw_data[:, :, i]
        transformed = compress(subject_data, level, threshold, wavelet)
        if num == 0:
            transformed_array = transformed[0]
        else:
            transformed_array = np.vstack(
                (transformed[num][0], transformed[num][1], transformed[num][2]))
        # transformed_array = pywt.coeffs_to_array(transformed)[0] # all coefficients
        # transformed_array = transformed[0] # approximation coefficients

        features_arr.append(transformed_array.flatten())

    features_arr = np.array(features_arr)
    features_arr = (features_arr - features_arr.min()) / \
        (features_arr.max() - features_arr.min())
    return features_arr


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
        features = create_features(data, params["level"],
                                   params["threshold"], params["wavelet"], params["coefficient_number"])
        combined = np.column_stack((features, all_labels))
        model_data.append(combined)
else:
    for i, f in enumerate(tqdm(os.listdir(input_folder))):
        data = np.load(os.path.join(input_folder, f))
        features = create_features(data, params["level"],
                                   params["threshold"], params["wavelet"], params["coefficient_number"])
        combined = np.column_stack((features, all_labels[i]))
        model_data.append(combined)

model_data = np.vstack(model_data)
np.save(os.path.join(output_folder, "wavelet_features"), model_data)
