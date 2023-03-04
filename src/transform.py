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

# Test data set split ratio
shape = params["shape"]

input_folder = os.path.join(sys.argv[1], "raw")
output_folder = os.path.join(sys.argv[2])

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

print("Transforming data...")
for file in tqdm(os.listdir(input_folder)):
    mat_file = scipy.io.loadmat(os.path.join(input_folder, file))
    data = mat_file['EEG']['data'][0][0].reshape(shape[0], shape[1], shape[2])
    np.save(os.path.join(output_folder, file.split('.')[0]), data)
print("Done.")
