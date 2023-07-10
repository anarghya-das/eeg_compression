import pandas as pd
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
import os
import sys
import yaml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pickle
import numpy as np
from tqdm import tqdm

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
print(features.shape)
X_train, X_test, Y_train, Y_test = train_test_split(
    features[:, :-1], features[:, -1], train_size=train_size, test_size=test_size, random_state=state)

# model = svm.SVC(kernel=svm_kernel, C=svm_C,
#                 degree=svm_degree, random_state=state)

# model = GradientBoostingClassifier(
#     n_estimators=1000, random_state=state)

# model.fit(X_train, Y_train)
# print(metrics.accuracy_score(Y_train, model.predict(X_train)))
# print(metrics.accuracy_score(Y_test, model.predict(X_test)))


bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X_train, Y_train)
# dfscores = pd.DataFrame(fit.scores_)
# dfcolumns = pd.DataFrame(X_train.columns)
# scores = pd.concat([dfcolumns, dfscores], axis=1)
# scores.columns = ['Specs', 'Score']
print(fit.scores_)

# models = [
#     ('LogReg', LogisticRegression(max_iter=1000)),
#     ('RF', RandomForestClassifier()),
#     ('SVM', SVC()),
#     ('GNB', GaussianNB()),
#     ('XGB', XGBClassifier()),
#     ("DecisionTree", DecisionTreeClassifier()),
#     ("MLPClassifier", MLPClassifier(random_state=state)),
#     ("GradientBoostingClassifier", GradientBoostingClassifier(loss='exponential',
#                                                               n_estimators=1000, random_state=state))
# ]

# results = []
# names = []
# scoring = ['accuracy', 'precision_weighted',
#            'recall_weighted', 'f1_weighted', 'roc_auc']
# for name, model in tqdm(models):
#     kfold = model_selection.KFold(
#         n_splits=10, random_state=state, shuffle=True)
#     cv_results = model_selection.cross_validate(
#         model, X_train, Y_train, cv=kfold, scoring=scoring)
#     clf = model.fit(X_train, Y_train)
#     y_pred = clf.predict(X_test)
#     print(name)
#     print(classification_report(
#         Y_test, y_pred, target_names=["Left", "Right"]))

# results.append(cv_results)
# names.append(name)
