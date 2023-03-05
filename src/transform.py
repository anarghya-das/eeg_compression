import os
import sys
import yaml
import scipy
import numpy as np
from tqdm import tqdm

params = yaml.safe_load(open("params.yaml"))["transform"]

if len(sys.argv) != 3:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython prepare.py data-folder output-folder\n")
    sys.exit(1)

data_name = sys.argv[1].split("/")[-1]
# Test data set split ratio
shape = params["datasets"][data_name]["shape"]

input_folder = os.path.join(sys.argv[1], "raw")
output_folder = os.path.join(sys.argv[2])

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

print("Transforming data...")
for file in tqdm(os.listdir(input_folder)):
    mat_file = scipy.io.loadmat(os.path.join(input_folder, file))
    if data_name == "mi":
        train = np.reshape(mat_file['x_train'], (mat_file['x_train'].shape[1],
                                                 mat_file['x_train'].shape[0], mat_file['x_train'].shape[2]))
        test = np.reshape(mat_file['x_test'], (mat_file['x_test'].shape[1],
                                               mat_file['x_test'].shape[0], mat_file['x_test'].shape[2]))
        data = np.concatenate((train, test), axis=2)
        labels = mat_file['y_train']
    elif data_name == "seed-vig":
        data = mat_file['EEG']['data'][0][0].reshape(
            shape[0], shape[1], shape[2])
    np.save(os.path.join(output_folder, file.split('.')[0]), data)
print("Done.")
