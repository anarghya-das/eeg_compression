import os
import sys
import yaml
from sklearn.model_selection import train_test_split

from tensorflow import keras
from keras import layers

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
test_size = 1 - train_size

features = np.load(features_path)

model = keras.Sequential(
    [
        layers.Dense(128, activation="relu"),
        layers.Dense(3, activation="softmax"),
    ]
)

X_train, _, Y_train, _ = train_test_split(
    features[:, :-1], features[:, -1], train_size=train_size, test_size=test_size, random_state=state)
print(np.max(Y_train))
model.compile(
    optimizer=keras.optimizers.Adam(), 
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy'],
)

model.fit(
    X_train,
    Y_train,
    batch_size=16,
    epochs=12,
)

model.save('mdl.h5')