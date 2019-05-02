import time
from collections import defaultdict

import pandas as pd
from sklearn.model_selection import StratifiedKFold

import vul_common
# d: Number of features
# n(i): Number of samples for project i
from vul_common import number_of_folds, features_cols, label_col

main_start = time.time()
skf = StratifiedKFold(n_splits=number_of_folds, shuffle=True, random_state=vul_common.random_state)

project_dfs = {}
project_features = {}  # <Project Name, Features of Project>
project_labels = {}  # <Project Name, Labels of Project>
project_fold_indexes = defaultdict(list)  # <Project Name, [(Train Fold Indexes, Test Fold Indexes)]>
file = "out/test.csv"
pd.read_csv(file)
for project_name in vul_common.projects_names:
    filename = vul_common.table_csv_filename(project_name)
    project_df = pd.read_csv(filename)
    # If a row represents the state of file before a fix than it is vulnerable 1, otherwise neutral (fixed) 0
    project_df['Occurrence'] = project_df['Occurrence'].map({'before': 1, 'after': 0})
    # Change the name of label column because why not
    project_df.rename(index=str, columns={"Occurrence": label_col}, inplace=True)
    project_dfs[project_name] = project_df
    features = project_df[features_cols]  # A n(i) * d matrix
    labels = project_df[label_col]  # A n(i) vector
    project_features[project_name] = features
    project_labels[project_name] = labels
    for train_index, test_index in skf.split(features, labels):
        project_fold_indexes[project_name].append((train_index, test_index))

for project_name in vul_common.projects_names:
    for fold_no in range(number_of_folds):
        train_index = project_fold_indexes[project_name][fold_no][0]
        test_index = project_fold_indexes[project_name][fold_no][1]
        train_of_fold = project_features[project_name].iloc[train_index, :]
        test_of_fold = project_features[project_name].iloc[test_index, :]
        train_of_fold[label_col] = project_labels[project_name][train_index]
        test_of_fold[label_col] = project_labels[project_name][test_index]
        train_of_fold.to_csv(vul_common.metrics_train_fold_csv_filename(project_name, fold_no))
        test_of_fold.to_csv(vul_common.metrics_test_fold_csv_filename(project_name, fold_no))

# skf.split(X, y)

main_time = time.time() - main_start
print("------> All tasks finished in {}".format(main_time))
