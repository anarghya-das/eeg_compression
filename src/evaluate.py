import os
import pickle
import sys
import numpy as np
from sklearn.model_selection import train_test_split
import yaml
from sklearn import metrics
from dvclive import Live
from matplotlib import pyplot as plt
import pandas as pd

params = yaml.safe_load(open("params.yaml"))["evaluate"]

if len(sys.argv) != 3:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython evaluate.py model features\n")
    sys.exit(1)

model_file = sys.argv[1]
features_path = os.path.join(sys.argv[2], "wavelet_features.npy")
features = np.load(features_path)
train_size = params['split']
state = params['state']
test_size = 1 - train_size

_, X_test, _, Y_test = train_test_split(
    features[:, :-1], features[:, -1], train_size=train_size, test_size=test_size, random_state=state)

with open(model_file, "rb") as fd:
    model = pickle.load(fd)

Y_pred = model.predict(X_test)
accuracy = metrics.accuracy_score(Y_test, Y_pred)
f1 = metrics.f1_score(Y_test, Y_pred, average='weighted')

with Live("evaluation") as live:
    live.log_metric("accuracy", accuracy)
    live.log_metric("f1", f1)
    live.log_sklearn_plot("confusion_matrix", Y_test, Y_pred)

    # fig, axes = plt.subplots(dpi=100)
    # fig.subplots_adjust(bottom=0.2, top=0.95)
    # importances = model.feature_importances_
    # forest_importances = pd.Series(
    #     importances).nlargest(n=30)
    # axes.set_ylabel("Mean decrease in impurity")
    # forest_importances.plot.bar(ax=axes)
    # fig.savefig(os.path.join("evaluation", "plots", "importance.png"))
