import os
import sys
import yaml
from sklearn.model_selection import train_test_split
from sklearn import svm
import pickle
import numpy as np

params = yaml.safe_load(open("params.yaml"))["train"]

if len(sys.argv) != 2:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write(
        "\tpython featurization.py dataset-dir-path\n")
    sys.exit(1)

features_path = os.path.join(sys.argv[1], "wavelet_features.npy")
train_size = params['split']
state = params['state']
svm_kernel = params['svm_kernel']
svm_C = params['svm_C']
svm_degree = params['svm_degree']
test_size = 1 - train_size

features = np.load(features_path)
X_train, _, Y_train, _ = train_test_split(
    features[:, :-1], features[:, -1], train_size=train_size, test_size=test_size, random_state=state)

svm = svm.SVC(kernel=svm_kernel, C=svm_C,
              degree=svm_degree, random_state=state)
svm.fit(X_train, Y_train)
pickle.dump(svm, open('model.pkl', 'wb'))
